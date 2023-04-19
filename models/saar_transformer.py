import torch
from torch import Tensor, nn
import torch.optim as optim
import torch.nn.functional as F

import lightning.pytorch as pl
from lion_pytorch.lion_pytorch import Lion

from typing import Dict, Iterable, Optional
import numpy as np

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def init_transformer(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        torch.nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)

class MultiHeadAttention(pl.LightningModule):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        causal = False,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        # watch out, the returned qk is not valid
        wv, qk = self.qkv_attention(q, k, v, causal)
                
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, causal = False
    ):
        n_batch, n_ctx, n_state = q.shape
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        wv = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=causal)

        # we've returned q@k which we don't have now, but it's not used so just let's keep two
        # return values
        return wv.permute(0, 2, 1, 3).flatten(start_dim=2), None

class ResidualAttentionBlock(pl.LightningModule):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        causal = False,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), causal=causal, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x
    
# encoder model, accepts 1500 semantic tokens
class SEncoder(pl.LightningModule):
    def __init__(self, sin_embs, depth=6, length=1500, width=384, S_codes=1024, n_head=6, unique_Stoks=False):
        super(SEncoder, self).__init__()
    
        # embed semantic tokens
        if unique_Stoks:
            S_codes += 1 # for padding
        self.embedding = nn.Embedding(S_codes, width)
        self.register_buffer("positional_embedding", sin_embs)

        self.layers = nn.Sequential(*[
            ResidualAttentionBlock(width, n_head) for _ in range(depth)
        ])

        self.ln_post = LayerNorm(width)
        
    def forward(self, Stoks):
        xin = self.embedding(Stoks)
        
        assert xin.shape[1:] == self.positional_embedding.shape, "incorrect semantic token shape"
        xin = (xin + self.positional_embedding).to(xin.dtype)

        return self.ln_post(self.layers(xin))

# AR decoder, accepts and outputs interleaved acoustic tokens (1024 is a start of sequence token)
class ADecoder(pl.LightningModule):
    def __init__(self, sin_embs, depth=6, length=4500, width=384, A_codes=1024, n_head=6):
        super(ADecoder, self).__init__()
    
        # embed semantic tokens
        self.embedding = nn.Embedding(A_codes+1, width)
        self.register_buffer("positional_embedding", sin_embs)
        
        # before adding the encoder features
        self.layers = nn.ModuleList([
            ResidualAttentionBlock(width, n_head) for _ in range(depth)
        ])

        # after adding the encoder features
        self.layers2 = nn.ModuleList([
            ResidualAttentionBlock(width, n_head) for _ in range(depth)
        ])

        self.ln_post = LayerNorm(width)
        
    def forward(self, Atoks, xenc):
        sot = self.embedding(torch.tensor([1024]).cuda()).repeat(Atoks.shape[0],1,1)
        if Atoks.shape[-1] > 0:
            if Atoks.shape[-1] > 4499:
                Atoks = Atoks[:,:-1]
            Aembs = self.embedding(Atoks)
            Aembs = torch.cat([sot, Aembs], dim=-2)
        else:
            Aembs = sot

        xin = (Aembs + self.positional_embedding[:Aembs.shape[1]]).to(xenc.dtype)
    
        x = xin

        for l in self.layers: x = l(x, causal=True)
            
        x += xenc.repeat_interleave(3, dim=-2)[:,:Aembs.shape[1]]

        for l in self.layers2: x = l(x, causal=True)
        
        x = self.ln_post(x)
        
        logits = (x @ self.embedding.weight.to(x.dtype).T).float()
        return logits

class SAARTransformer(pl.LightningModule):
    def __init__(self, width=384, depth=2, n_head=6, unique_Stoks=False, train_loader=None, model_hparams=None):
        super(SAARTransformer, self).__init__()

        # generate positional embeddings and subsample for the encoder, so they stay compatible
        pos_embs = sinusoids(4500, width)
        
        self.encoder = SEncoder(pos_embs[::3], width=width, n_head=n_head, depth=depth, unique_Stoks=unique_Stoks)
        self.decoder = ADecoder(pos_embs, width=width, n_head=n_head, depth=depth)
        self.train_loader = train_loader

        self.model_hparams = model_hparams
        
        self.apply(init_transformer)

    def forward(self, Stoks, Atoks, loss=True):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        xenc = self.encoder(Stoks.to(torch.long))
        logits = self.decoder(Atoks, xenc)
        if loss is not None:
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), Atoks.view(-1))
        return logits, loss
    
    def configure_optimizers(self):
        """ Initialize Lion optimizer"""
        all_params = set(self.parameters())
        wd_params = set()
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                wd_params.add(m.weight)
                if m.bias is not None:
                    wd_params.add(m.bias)
        no_wd_params = all_params - wd_params

        optimizer = Lion(
            lr=self.model_hparams['lr0'] * self.model_hparams['warmup_mul'],
            params=[
                {"params": list(wd_params), "weight_decay": self.model_hparams['weight_decay']},
                {"params": list(no_wd_params), "weight_decay": 0.0}]
        )

        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            pct_start=self.model_hparams['pct_start'],
            max_lr=self.model_hparams['lr0'],
            steps_per_epoch=len(self.train_loader),
            epochs=self.model_hparams['epochs']
        )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        train_logits, train_loss = self.forward(x, y)

        self.log("train_loss", train_loss, sync_dist=True)
        return train_loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        val_logits, val_loss = self.forward(x, y)

        self.log("val_loss", val_loss, sync_dist=True)
        return val_loss
    
    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        test_logits, test_loss = self.forward(x, y)

        self.log("test_loss", test_loss, sync_dist=True)
        return test_loss