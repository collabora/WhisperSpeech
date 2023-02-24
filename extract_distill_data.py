import torch
import torchaudio
import sys
from pathlib import Path
from fastprogress import progress_bar, master_bar
import numpy as np

import whisper
whmodel = whisper.load_model('tiny.en')
tokenizer = whisper.tokenizer.get_tokenizer(False, language='en')

datadir = Path('/mnt/small/')

spkid = sys.argv[1]

def load(fname, newsr=24000):
    x, sr = torchaudio.load(fname)
    _tform = torchaudio.transforms.Resample(sr, newsr)
    return _tform(x).cuda().unsqueeze(0)

# same as above but rolled into a function
def encode_semantic_logits(fname):
    audio = load(fname, newsr=whisper.audio.SAMPLE_RATE)
    mel = whisper.log_mel_spectrogram(audio[0,0])
    init_tokens = torch.tensor([tokenizer.sot_sequence]).repeat(whisper.audio.N_FRAMES, 1).cuda()
    embs = []
    toks = []
    for start in range(0, mel.shape[-1], whisper.audio.N_FRAMES):
        sample = mel[:,start:]
        with torch.no_grad():
            padded = whisper.audio.pad_or_trim(sample, whisper.audio.N_FRAMES).unsqueeze(0)
            emb = whmodel.encoder(padded)
            tokens = whmodel.decode(padded, whisper.DecodingOptions(language='en'))[0].tokens
            embs.append(emb.cpu())
            toks.append(tokens)
    return torch.stack(embs, axis=0), toks

outdir = Path(f'whisper-tiny-decoder-{spkid}')
outdir.mkdir(exist_ok=True)
for name in progress_bar(list((datadir/str(spkid)).rglob('*.flac'))):
    embs, toks = encode_semantic_logits(name)
    torch.save(embs.to(torch.float16), outdir/name.with_suffix('.whisper').name)
    torch.save(toks, outdir/name.with_suffix('.tokens').name)
print()
