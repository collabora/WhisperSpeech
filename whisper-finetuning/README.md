# Whisper Encoder/Decoder training

- Freeze decoder weights and train the encoder from scratch:
```bash
 python train.py --freeze_decoder
```

- Freeze encoder weights and train the decoder from scratch:
```bash
 python train.py --freeze_encoder
```

- Results for training the whisper decoder from scratch on 360 hour ```librispeech_asr``` are [here on huggingface hub](https://huggingface.co/makaveli10/whisper-tiny-decoder-libriasr-clean-360h/tensorboard)