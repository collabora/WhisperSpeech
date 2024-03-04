'''
DESCRIPTION~

Processes a body of text directly into audio playback using the sounddevice library.

PLEASE NOTE~

If you need more granular control, such as being able to process sentences in one thread (one sentence at a time) while simultaneously playing them in another thread (reducing latency), consult the "text_to_audio_playback.py" example.  It uses the "generate" method in conjunction with the "sounddevice" library directly.

This example uses the "genereate_to_playback" method instead, which is good for reducing the length of your script, especially with shorter passages where latency is not as important.

INSTALLATION INSTRUCTIONS~

(1)  create a virtual environment and activate it
(2)  install pytorch by going to the following website and running the appropriate command for your platform and setup:

https://pytorch.org/get-started/locally/
---This script has been tested up to Torch 2.2.0.

(3)  pip3 install WhisperSpeech
(4)  pip3 install sounddevice==0.4.6
(5)  python text_to_playback.py
'''

from whisperspeech.pipeline import Pipeline

# pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')
# pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')
pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-base-en+pl.model')

pipe.generate_to_playback("""
 This is some sample text. You would add text here that you want spoken and then only leave one of the above lines uncommented for the model you want to test. This text is being used to test a new generate to playback method within the pipeline script.  It would require adding sounddevice as a dependency since that's what performs the playback.
""")