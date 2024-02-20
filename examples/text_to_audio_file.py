'''
Simple example that speaks some text and creates an audio file named output_audio.wav in the same directory in which the script is run.

INSTALLATION INSTRUCTIONS~
**Tested on Windows

(1)  create a virtual environment and activate it
(2)  install pytorch by going to the following website and running the appropriate command for your platform and setup:

https://pytorch.org/get-started/locally/

(3)  pip3 install WhisperSpeech
(4)  pip3 install pydub==0.25.1
(5)  python text_to_audio_file.py
'''

from pydub import AudioSegment
import numpy as np
from whisperspeech.pipeline import Pipeline

# pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')
# pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')
pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-base-en+pl.model')

audio_tensor = pipe.generate("""
 This is some sample text.  You would add text here that you want spoken and then only leave one of the above lines uncommented for the model you want to test.  Note that this script does not rely on the standard method within the whisperspeech pipeline.  Rather, it replaces a part of the functionality with reliance on pydub instead.  This approach "just worked."  Feel free to modify or distribute at your pleasure.
""")

# generate uses CUDA if available; therefore, it's necessary to move to CPU before converting to NumPy array
audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)

if len(audio_np.shape) == 1:
    audio_np = np.expand_dims(audio_np, axis=0)
else:
    audio_np = audio_np.T

print("Array shape:", audio_np.shape)
print("Array dtype:", audio_np.dtype)

try:
    audio_segment = AudioSegment(
        audio_np.tobytes(), 
        frame_rate=24000, 
        sample_width=2, 
        channels=1
    )
    audio_segment.export('output_audio.wav', format='wav')
    print("Audio file generated: output_audio.wav")
except Exception as e:
    print(f"Error writing audio file: {e}")