# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/D. Common dataset utilities.ipynb.

# %% auto 0
__all__ = ['shard_glob', 'join_datasets', 'resampler', 'derived_name', 'derived_dataset', 'merge_in', 'AtomicTarWriter',
           'readlines']

# %% ../nbs/D. Common dataset utilities.ipynb 1
import os
import torch
import torchaudio
from pathlib import Path
import webdataset as wds
from contextlib import contextmanager

import torch.nn.functional as F

# %% ../nbs/D. Common dataset utilities.ipynb 2
def shard_glob(input):
    if '{' in input:
        return wds.shardlists.expand_urls(input)
    if isinstance(input, (Path, str)):
        path = Path(input)
        if path.is_dir():
            glob = '*.tar.gz'
        else:
            glob = path.name
            path = path.parent
        input = Path(path).glob(glob)
    else:
        raise ArgumentError("input should be either a list or a path with an optional glob specifier")
    return [str(x) for x in input]

# %% ../nbs/D. Common dataset utilities.ipynb 3
class join_datasets(torch.utils.data.IterableDataset):
    def __init__(self, datasets):
        self.datasets = datasets
        
    def __iter__(self):
        probs = torch.tensor([getattr(ds, 'weight', 1) for ds in self.datasets], dtype=torch.float)
        its = [iter(ds) for ds in self.datasets]
        while True:
            try:
                yield next(its[torch.multinomial(probs, 1)])
            except StopIteration:
                return    
    
    def __len__(self):
        return sum([ds.total_samples for ds in self.datasets])

# %% ../nbs/D. Common dataset utilities.ipynb 5
def resampler(newsr = 24000, key = 'samples_24k'):
    _last_sr = None
    tform = None
    
    def _resample(samples):
        for s in samples:
            sr = s['sample_rate']
            if sr != newsr:
                if sr != _last_sr: tform = torchaudio.transforms.Resample(sr, newsr)
                s[key] = tform(s['samples'])
            else:
                s[key] = s['samples']
            yield s
    
    return _resample

# %% ../nbs/D. Common dataset utilities.ipynb 6
def derived_name(input, kind, base="audio", suffix=".gz", dir=None):
    dir = Path(dir) if dir else Path(input).parent
    return str(dir/(Path(input).name.replace(f"-{base}-", f"-{kind}-") + suffix))

# %% ../nbs/D. Common dataset utilities.ipynb 7
def derived_dataset(kind, base='audio', suffix=".gz", decoders=[], dir=None):
    def deriver(url):
        url = str(derived_name(url, kind, base=base, suffix=suffix, dir=dir))
        return wds.WebDataset(
            wds.SimpleShardList([url])
        ).decode(*decoders)
    return deriver

# %% ../nbs/D. Common dataset utilities.ipynb 8
def merge_in(dataset_fun):
    """Merge a dataset into the current one returning samples with the union of keys. Pass in a function
    that takes a URL of a sample and returns a dataset for it (called everytime the URL changes).
    
    It requires (and validates) that both datasets have the same ordering of keys so you have
    to use it before any sample shuffling. Shard shuffling is ok.
    """
    def merge_loop(main_samples):
        #print("new merge loop:", dataset_fun)
        merged_samples = None
        cur_url = None
        i = None
        for s in main_samples:
            url = s['__url__']
            if url != cur_url:
                # this will open a new file when we get the first sample with a new __url__
                merged_samples = iter(dataset_fun(url))
                cur_url = url
            try:
                merge_s = next(merged_samples)
            except StopIteration:
                # if the original shard got repeated we won't observe a __url__ change
                # in this case restart the dataset from the beginning
                merged_samples = iter(dataset_fun(url))
                merge_s = next(merged_samples)
            assert merge_s['__key__'] == s['__key__'], f"sample keys don't match: {merge_s['__key__']}, {s['__key__']} in file {s['__url__']}"
            news = {}
            news.update(merge_s)
            news.update(s)
            yield news
    return merge_loop

# %% ../nbs/D. Common dataset utilities.ipynb 9
def split_to_chunks(stream, ikey='vad.npy', metakeys=[], pad_to_seconds=30, random_shift=False):
    for s in stream:
        audio, sr = s['audio']
        imax = len(s[ikey]) - 1
        for i,(ts,te) in enumerate(s[ikey]):
            samples = audio[0,int(ts*sr):int(te*sr)]
            if pad_to_seconds is not None:
                padding = pad_to_seconds*sr-samples.shape[-1]
                lpad = random.randint(0, padding) if random_shift else 0
                samples = F.pad(samples, (lpad, padding-lpad))
            subs = {"__key__": s['__key__'] + f"_{i:03d}",
                    "src_key": s['__key__'],
                    "__url__": s['__url__'],
                    "i": i, "imax": imax,
                    "tstart": ts, "tend": te, "total_seconds": audio.shape[-1]/sr,
                    "lpad": lpad, "rpad": padding-lpad,
                    "lpad_s": lpad/sr, "rpad_s": (padding-lpad)/sr,
                    "samples": samples, "sample_rate": sr}
            for k in metakeys:
                subs[k] = s[k][i]
            yield subs

# %% ../nbs/D. Common dataset utilities.ipynb 10
import re
import tempfile

# %% ../nbs/D. Common dataset utilities.ipynb 11
def torch_audio_opus(key, data):
    """Decode audio using the torchaudio library.

    :param key: file name extension
    :param data: data to be decoded
    """
    extension = re.sub(r".*[.]", "", key)
    if extension not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma", "opus"]:
        return None

    import torchaudio

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
        return torchaudio.load(fname)

# %% ../nbs/D. Common dataset utilities.ipynb 12
def find_audio(stream, okey='audio', ikeys='flac;mp3;wav;ogg;opus'):
    ikeys = ikeys.split(';')
    for s in stream:
        for ikey in ikeys:
            if ikey in s:
                s[okey] = s[ikey]
                yield s
                break
            # implicitly skips elements without any audio

# %% ../nbs/D. Common dataset utilities.ipynb 13
def vad_dataset(shards, ikey='vad.npy', kind='vad'):
    return wds.WebDataset(shards).compose(
        wds.decode(torch_audio_opus),
        merge_in(derived_dataset(kind)),
        find_audio,
        lambda x: split_to_chunks(x, ikey=ikey),
    )

# %% ../nbs/D. Common dataset utilities.ipynb 14
@contextmanager
def AtomicTarWriter(name, throwaway=False):
    tmp = name+".tmp"
    with wds.TarWriter(tmp, compress=name.endswith('gz')) as sink:
        yield sink
    if not throwaway:
        os.rename(tmp, name)

# %% ../nbs/D. Common dataset utilities.ipynb 15
def readlines(fname):
    with open(fname) as file:
        return [line.rstrip() for line in file]
