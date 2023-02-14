# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/1. Acoustic token extraction.ipynb.

# %% auto 0
__all__ = ['load', 'load_model', 'extract_Atoks', 'extract_acoustic']

# %% ../nbs/1. Acoustic token extraction.ipynb 2
import torch
import torchaudio
import gc

from pathlib import Path
from fastcore.script import *
from fastprogress import progress_bar, master_bar
from utils import get_compute_device

compute_device = get_compute_device()

# %% ../nbs/1. Acoustic token extraction.ipynb 5
def load(fname, newsr=24000):
    """Load an audio file to the GPU and resample to `newsr`."""
    x, sr = torchaudio.load(fname)
    _tform = torchaudio.transforms.Resample(sr, newsr)
    return _tform(x).to(compute_device).unsqueeze(0)

# %% ../nbs/1. Acoustic token extraction.ipynb 6
def load_model():
    "Load the pretrained EnCodec model"
    from encodec.model import EncodecModel
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(1.5)
    model.to(compute_device).eval();
    return model

# %% ../nbs/1. Acoustic token extraction.ipynb 7
def extract_Atoks(model, audio):
    """Extract EnCodec tokens for the given `audio` tensor (or file path)
    using the given `model` (see `load_model`)."""
    if isinstance(audio, (Path, str)):
        audio = load(audio)
    with torch.no_grad():
        frames = torch.cat([model.encode(segment)[0][0]
                            for segment in torch.split(audio, 320*20000, dim=-1)], dim=-1)
    return frames

# %% ../nbs/1. Acoustic token extraction.ipynb 8
@call_parse
def extract_acoustic(
        srcdir:Path,  # source dir, should contain *.flac files
        outdir:Path,  # output dir, will get the *.encodec files
    ): 
    "Convert audio files to .encodec files with tensors of tokens"
    model = load_model()
    outdir.mkdir(exist_ok=True, parents=True)
    for name in progress_bar(list(srcdir.rglob('*.flac'))):
        outname = outdir/name.with_suffix('.encodec').name
        tokens = extract_Atoks(model, name)
        torch.save(tokens, outname)
        del tokens
        gc.collect()
