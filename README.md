[![](https://dcbadge.vercel.app/api/server/FANw4rHD5E)](https://discord.gg/FANw4rHD5E)  
*If you have questions or you want to help you can find us in the #audio-generation channel on the LAION Discord server.*

# SPEAR-TTS

An unofficial PyTorch implementation of [SPEAR-TTS](https://google-research.github.io/seanet/speartts/examples/).

We are not targeting an exact copy – to speed up training we want to use existing Open Source models as bases:
[Whisper](https://github.com/openai/whisper) encoder to generate semantic tokens and [EnCodec](https://github.com/facebookresearch/encodec) for acoustic modeling.

Following Google Brain we'll train on the LibriLight dataset. Ultimately
we want to target multiple languages (Whisper and EnCodec are both multilanguage).

## Progress updates

**UPDATE 2023-04-13**: We have trained a preliminary T->S model and a new 3kbps S->A model which improves the speech quality. Both models are far from perfect yet but we are clearly moving in the right direction (to the moon 🚀🌖!).

End-to-end TTS model with ≈ 6% WER (both T->S and S->A sampled with simple multinomial sampling at T = 0.7, no beam search) see https://github.com/collabora/spear-tts-pytorch/issues/9 for more details:

(don't forget to unmute the video)

https://user-images.githubusercontent.com/107984/231755045-e7d55a7a-6d97-4a0a-a8cf-1bc7f54c9217.mp4

Ground truth:

https://user-images.githubusercontent.com/107984/231755210-7150636b-18c2-4dff-a8f4-9db0b932ad5f.mp4

**UPDATE 2023-04-03**: We have trained a working S->A model. It does not sound amazing but that is mostly because of EnCodec quality at 1.5kbps.

Validation set ground truth (don't forget to unmute):

https://user-images.githubusercontent.com/107984/229439299-3aca954c-f044-4270-a4e5-4f847fd5d929.mov

The generated output from the S->A model (multinomial sampling, temperature 0.8):

https://user-images.githubusercontent.com/107984/229439418-92575be4-a892-40bb-97f7-5bfda5b2bf1d.mov

## Roadmap

- [x] [Extract acoustic tokens](https://github.com/collabora/spear-tts-pytorch/issues/2)
- [x] [Extract Whisper embeddings and quantize them to semantic tokens](https://github.com/collabora/spear-tts-pytorch/issues/3)
- [x] [Semantic token to acoustic token (S->A) model](https://github.com/collabora/spear-tts-pytorch/issues/4)
- [ ] [Text token to semantic token (T->S) model](https://github.com/collabora/spear-tts-pytorch/issues/9)
- [ ] [Improve the EnCodec speech quality](https://github.com/collabora/spear-tts-pytorch/issues/10)
- [ ] [Gather a bigger emotive speech dataset](https://github.com/collabora/spear-tts-pytorch/issues/11)
- [ ] [Train final high-quality models](https://github.com/collabora/spear-tts-pytorch/issues/12)

## Architecture

### Whisper for modeling semantic tokens

![Using Whisper for semantic token extraction diagram](whisper-block.png)

Pros:
 
 - Whisper training should be a lot better at extracting semantic information than a masked language model with
   contrastive loss (w2v-BERT)
 - it's pretrained on 600k hours of multilingual speech (vs. 60k for w2v-BERT used in the paper)
 - freely available

Cons:

 - 2x higher "symbol rate" (50 vec/s) than w2v-BERT (25 vec/s) which means training the semantic->acoustic transformer
   may take longer (this turned out not to matter in practice – there are only 1500 semantic tokens for 30 seconds of audio vs. 4500 acoustic tokens)

### EnCodec for modeling acoustic tokens

![EnCodec block diagram](https://github.com/facebookresearch/encodec/raw/main/architecture.png)

Pros:

 - High-quality pretrained model is available

Cons:

 - Comparing the speech samples with SPEAR-TTS, EnCodec needs 6kbps to get the same quality
   (SoundStream retrained only on speech seems to work with 1.5kbps)
 - CC-BY-NC license

We may switch to the [OpenSource SoundStream re-implementation](https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/soundstream.py) or train a new speech-only model.

## Appreciation

[<img height=80 src="https://user-images.githubusercontent.com/107984/229537027-a6d7462b-0c9c-4fd4-b69e-58e98c3ee63f.png" alt="Collabora logo">](https://www.collabora.com)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[<img height=80 src="https://user-images.githubusercontent.com/107984/229535036-c741d775-4a9b-4193-89a0-9ddb89ecd011.png" alt="LAION logo">](https://laion.ai)

This work would not be possible without the generous sponsorships from:

- [Collabora](https://www.collabora.com) – code development and model training
- [LAION](https://laion.ai) – community building and datasets

We are available to help you with both Open Source and proprietary AI projects. You can reach us via the Collabora website or on Discord ([![](https://dcbadge.vercel.app/api/shield/270267134960074762?style=flat)](https://discordapp.com/users/270267134960074762) and [![](https://dcbadge.vercel.app/api/shield/1088938086400016475?style=flat)](https://discordapp.com/users/1088938086400016475))

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
