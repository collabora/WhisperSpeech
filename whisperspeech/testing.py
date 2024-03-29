# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/C2. Testing.ipynb.

# %% auto 0
__all__ = ['test_model']

# %% ../nbs/C2. Testing.ipynb 1
import webdataset as wds

# %% ../nbs/C2. Testing.ipynb 2
def test_model(model, ds, bs=1):
    dev = next(model.parameters()).device
    logits, loss = model(*[x.to(dev) for x in next(iter(wds.WebLoader(ds, batch_size=None).unbatched().batched(bs)))])
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            print('Unused parameter: '+name)
