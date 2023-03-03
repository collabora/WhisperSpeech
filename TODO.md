# Things to fix

- encodec splits audio into chunks but does not have any overlap, is that correct?
- Whisper text token decoding for distillation returns a bit different tokens than the output of the official decoder code
- Whisper sembs are extracted without any overlap
