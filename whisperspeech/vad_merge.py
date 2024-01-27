# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/1C. VAD merging.ipynb.

# %% auto 0
__all__ = ['derived_name']

# %% ../nbs/1C. VAD merging.ipynb 2
import random

import numpy as np
import torch
import torch.nn.functional as F

from fastprogress import progress_bar
from fastcore.script import *

from . import utils
import webdataset as wds

# %% ../nbs/1C. VAD merging.ipynb 4
def derived_name(input, kind, base="audio"):
    return input.replace(base, kind) + ".gz"

# %% ../nbs/1C. VAD merging.ipynb 8
# we need to split first to merge in the spk_emb.npy data
# this is similar to utils.split_to_chunks but works without the audio data
def split(stream, ikey='vad.npy'):
    empty = []
    for s in stream:
        imax = len(s[ikey]) - 1
        if len(s[ikey]) == 0:
            # Preserve info about audio files without any speech.
            # We need to push this info through a weird side-channel 
            # because we want to be able to a merge with naively
            # splitted data.
            empty.append({"__key__": s['__key__'] + "_none",
                   "src_key": s['__key__'],
                   "__url__": s['__url__']})
        for i,(ts,te) in enumerate(s[ikey]):
            yield {"__key__": s['__key__'] + f"_{i:03d}",
                   "src_key": s['__key__'],
                   "__url__": s['__url__'],
                   "i": i, "imax": imax,
                   "tstart": ts, "tend": te,
                   "empty": empty}
            empty = []

def merge_by_src_key(stream):
    ms = None
    for s in stream:
        # push accumulated data
        if ms and s['src_key'] != ms['__key__']:
            yield ms
            ms = None
        # push all empty files we might have lost
        for vs in s.get("empty", []):
            yield {
                "__url__": vs['__url__'],
                "__key__": vs['src_key'],
                "spk_emb.npy": [],
                "vad.npy": [],
            }
        # prepare a merged record for the new data
        if ms is None:
            ms = {
                "__url__": s['__url__'],
                "__key__": s['src_key'],
                "spk_emb.npy": [],
                "vad.npy": [],
            }
        ms["spk_emb.npy"].append(s["spk_emb.npy"])
        ms["vad.npy"].append([s['tstart'], s['tend']])
    yield ms

# %% ../nbs/1C. VAD merging.ipynb 11
def random_cutter(dur):
    if random.random() < 0.5:
        return dur > 30 * (random.random()*0.95+0.05)
    else:
        return dur > 30

def chunk_merger(stream, should_cut=lambda x: x > 30):
    for s in stream:
        segments, speakers = s['vad.npy'], s['spk_emb.npy']
        if len(segments) == 0:
            s['vad.npy'], s['spk_emb.npy'] = np.array([]), np.array([])
            yield s
            continue
        curr_start = segments[0][0]
        curr_end = 0
        curr_spk = None
        curr_chunks = []
        spk_acc = torch.tensor(speakers[0])
        spk_acc_N = 1
        merged = []
        merged_chunks = []
        merged_spk = []

        for (ts,te),new_spk in zip(segments, speakers):
            secs = te - ts
            new_spk = torch.tensor(new_spk)
            spk_change = False
            if curr_spk is not None:
                sim = F.cosine_similarity(curr_spk, new_spk, dim=0)
                spk_change = sim < 0.5 if secs > 2 else sim < 0.1
            if (spk_change or should_cut(te - curr_start)) and curr_end - curr_start > 0:
                merged.append((curr_start, curr_end))
                merged_spk.append(spk_acc / spk_acc_N)
                merged_chunks.append(curr_chunks)
                curr_start = ts
                spk_acc = new_spk
                curr_chunks = []
            curr_spk = new_spk
            if secs > 2:
                spk_acc += new_spk
                spk_acc_N += 1
            curr_end = te
            curr_chunks.append((ts, te))
        merged.append((curr_start, curr_end))
        merged_spk.append(spk_acc / spk_acc_N)
        merged_chunks.append(curr_chunks)
        s['vad.npy'], s['spk_emb.npy'] = np.array(merged), torch.stack(merged_spk).numpy()
        s['subvads.pyd'] = merged_chunks
        yield s

# %% ../nbs/1C. VAD merging.ipynb 17
@call_parse
def prepare_mvad(
    input:str,  # FLAC webdataset file path (or - to read the names from stdin)
    output:str=None, # output file name
    eqvad:bool=False, # make the chunk length distribution more uniform
):
    if eqvad:
        def merger(x):
            return chunk_merger(x, random_cutter)
        kind = 'eqvad'
    else:
        merger = chunk_merger
        kind = 'maxvad'
    
    ds = wds.WebDataset([utils.derived_name(input, 'vad')]).compose(
        wds.decode(),
        split,
        utils.merge_in(utils.derived_dataset('spk_emb', base='vad', suffix='')),
        merge_by_src_key,
        merger,
    )

    with utils.AtomicTarWriter(derived_name(input, kind)) as sink:
        for s in progress_bar(ds, total='noinfer'):
            sink.write(s)

# %% ../nbs/1C. VAD merging.ipynb 20
def chunked_audio_dataset(shards, kind='maxvad'):
    return wds.WebDataset(shards).compose(
        wds.decode(utils.torch_audio_opus),
        utils.merge_in(utils.derived_dataset(kind)),
        utils.find_audio,
        lambda x: utils.split_to_chunks(x, metakeys=['spk_emb.npy']),
    )
