## WhisperSpeech dataset Makefiles

We are moving towards a make-based dataset processing setup. We have a `Makefile.WhisperSpeech`
include file which contains all the rules for data processing and per-dataset Makefiles that
configure the dataset-specific options. There is also a global Makefile that just includes all
the per dataset makefiles.
