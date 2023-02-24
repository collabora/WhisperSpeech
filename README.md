# SPEAR-TTS

An unofficial PyTorch implementation of [SPEAR-TTS](https://google-research.github.io/seanet/speartts/examples/).

We are not targeting an exact copy – to speed up training we want to use existing Open Source models as bases:
[Whisper](https://github.com/openai/whisper) encoder to generate semantic tokens and [EnCodec](https://github.com/facebookresearch/encodec) for acoustic modeling.

Following Google Brain we'll train on the LibreLight and LibreTTS datasets. Ultimately
we want to target multiple languages (Whisper and EnCodec are both multilanguage).

UPDATE 2023-02-24: I think I finally figured out how to train the semantic encodings bottleneck. Check the issues for more detailed progress updates.

## Whisper for modeling semantic tokens

![Using Whisper for semantic token extraction diagram](whisper-block.png)

Pros:
 
 - Whisper training should be a lot better at extracting semantic information than a masked language model with
   contrastive loss (w2v-VERT)
 - it's pretrained on 600k hours of multilingual speech (vs. 60k for w2v-BERT used in the paper)
 - freely available

Cons:

 - 2x higher "symbol rate" (50 vec/s) than w2v-BERT (25 vec/s) which means training the semantic->acoustic transformer
   may take longer
 - it seems that we'll need 6x higher symbol rate if we want to quantize the embeddings effectively, OTOH maybe later modeling tasks will be easier?

## EnCodec for modeling acoustic tokens

![EnCodec block diagram](https://github.com/facebookresearch/encodec/raw/main/architecture.png)

Pros:

 - High-quality pretrained model

Cons:

 - Comparing the speech samples with SPEAR-TTS, EnCodec needs 6kbps to get the same quality
   (SoundStream retrained only on speech seems to work with 1.5kbps)
 - CC-BY-NC license

We may switch to the [OpenSource SoundStream re-implementation](https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/soundstream.py) or train a new speech-only model.

## Citations

```bibtex
@article{SpearTTS,
  title = {Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision},
  url = {https://arxiv.org/abs/2302.03540},
  author = {Kharitonov, Eugene and Vincent, Damien and Borsos, Zalán and Marinier, Raphaël and Girgin, Sertan and Pietquin, Olivier and Sharifi, Matt and Tagliasacchi, Marco and Zeghidour, Neil},
  publisher = {arXiv},
  year = {2023},
}
```

```bibtex
@article{Whisper
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  publisher = {arXiv},  
  year = {2022},
}
```

```bibtex
@article{EnCodec
  title = {High Fidelity Neural Audio Compression},
  url = {https://arxiv.org/abs/2210.13438},
  author = {Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  publisher = {arXiv},
  year = {2022},
}
```
