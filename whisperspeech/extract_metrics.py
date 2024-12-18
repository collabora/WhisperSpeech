# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/3B. Speech quality metrics extraction.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/3B. Speech quality metrics extraction.ipynb 2
import sys
import os
from os.path import expanduser
import itertools
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from fastprogress import progress_bar
from fastcore.script import *

from pyannote.audio import Model
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
from . import vq_stoks, utils, vad_merge
import webdataset as wds

from .inference import get_compute_device

# %% ../nbs/3B. Speech quality metrics extraction.ipynb 4
@call_parse
def prepare_metrics(
    input:str,  # audio file webdataset file path
    output:str, # output shard path
    n_samples:int=None, # process a limited amount of samples
    cache_dir: str = None
    
):
    device = get_compute_device()

    model = Model.from_pretrained(expanduser('~/.cache/brouhaha.ckpt'), strict=False, cache_dir=cache_dir)
    snr_pipeline = RegressiveActivityDetectionPipeline(segmentation=model).to(torch.device(device))
        
    total = n_samples if n_samples else 'noinfer'

    if total == 'noinfer':
        import math, time
        start = time.time()
        ds = wds.WebDataset([utils.derived_name(input, 'mvad')]).decode()
        total = math.ceil(sum([len(x[f'max.spk_emb.npy']) for x in ds]))
        print(f"Counting {total} batches: {time.time()-start:.2f}")
    
    ds = vad_merge.chunked_audio_dataset([input], 'max').compose(
        wds.to_tuple('__key__', 'rpad', 'gain_shift.npy', 'samples', 'sample_rate'),
    )

    dl = wds.WebLoader(ds, num_workers=1, batch_size=None)
    
    with utils.AtomicTarWriter(output, throwaway=n_samples is not None) as sink:
        for keys, rpad, gain_shift, samples, sr in progress_bar(dl, total=total):
            with torch.no_grad():
                snd = samples
                if rpad > 0: snd = snd[:-rpad]
                snd = (snd - gain_shift[1]) * gain_shift[0]
                snd = snd.unsqueeze(0).to(device)

                res = snr_pipeline({
                    "sample_rate": sr, "waveform": snd
                })

            s = {
                "__key__": keys,
                "snr_c50.npy": np.array([res['snr'].mean(), res['c50'].mean()])
            }
            sink.write(s)
        sys.stdout.write("\n")
