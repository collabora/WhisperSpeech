from pathlib import Path
import sys, os, argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # SpearTTS root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import pandas as pd

from models.saar_transformer import SAARTransformer

wandb_logger = WandbLogger(project="SpearTTS")
torch.set_float32_matmul_precision('medium')

class SADataset(torch.utils.data.Dataset):
    def __init__(self, data, unique=False):
        self.data = data
        self.unique = unique
        self.samples = [(i,j) for i,name in enumerate(data['stoks']) for j in range(torch.load(name).shape[0])]
    
    def __len__(self):
        return len(self.samples)
    
    def S_tokens(self):
        return len(self)*1500
    
    def hours(self):
        return len(self)*30/3600
    
    def __repr__(self):
        return f"Dataset: {len(self)} samples, {self.S_tokens()} Stokens, {self.hours():.1f} hours)"
    
    def __getitem__(self, idx):
        i,j = self.samples[idx]
        row = self.data.iloc[i]
        jA = j * 2250
        Stoks = torch.load(row['stoks'], map_location='cpu')[j]
        Atoks = torch.load(row['atoks'], map_location='cpu')[0,:,jA:jA+2250].T.reshape(-1)
        if self.unique:
            x = torch.unique_consecutive(Stoks)
            Stoks = F.pad(x, (0, Stoks.shape[0] - x.shape[0]), value=1024)
        return Stoks, F.pad(Atoks, (0, 4500 - len(Atoks)), value=1024)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--input-dir', type=str, default='', help='input data path')
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/", help="directory to save the checkpoints")
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='optimizer weight decay')
    parser.add_argument('--warmup-multiplier', type=float, default=1e-2, help='optimizer warmup multiplier')
    parser.add_argument('--lr0', type=float, default=1e-4, help='optimizer initial learning rate')
    parser.add_argument('--pct-start', type=float, default=0.3, help='optimizer percentage of total number of epochs when learning rate rises during one cycle')
    parser.add_argument('--depth', type=int, default=2, help='model depth')

    args = parser.parse_args().__dict__

    input_dir: str = args.pop("input_dir")
    checkpoint_dir: str = args.pop("checkpoint_dir")
    num_workers: int = args.pop("workers")
    batch_size: int = args.pop("batch_size")
    epochs: int = args.pop("epochs")

    hyp_params = {}
    hyp_params['warmup_mul'] = args['warmup_multiplier']
    hyp_params['pct_start'] = args['pct_start']
    hyp_params['weight_decay'] = args['weight_decay']
    hyp_params['lr0'] = args['lr0']
    hyp_params['epochs'] = epochs

    datadir = Path(input_dir)
    data = pd.read_feather('nbs/token-dataset.feather')
    atoks = {x.name:x for x in list(Path(datadir).rglob('*.encodec'))}
    stoks = {x.name:x for x in list(Path(datadir).rglob('*.stoks'))}
    data['atoks'] = data.apply(lambda x: str(atoks[Path(x['afile']).with_suffix('.encodec').name]), axis=1)
    data['stoks'] = data.apply(lambda x: str(stoks[Path(x['afile']).with_suffix('.stoks').name]), axis=1)

    data6454 = data[data['speaker'] == '6454']

    val_data, train_data = data6454[:12], data6454[12:]
    val_ds = SADataset(val_data, unique=False)
    print(val_ds)

    train_ds = SADataset(train_data, unique=False)
    print(train_ds)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="SAART-{epoch}-{step}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=4,
        every_n_epochs=1,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=False)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              drop_last=False,
                              shuffle=True)

    model = SAARTransformer(depth=args['depth'], train_loader=train_loader, model_hparams=hyp_params)

    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator="gpu",
                         profiler="simple",
                         precision='16-mixed',
                         enable_checkpointing=True,
                         logger=wandb_logger,
                         callbacks=[ckpt_callback, lr_monitor_callback])

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)