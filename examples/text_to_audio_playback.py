'''
DESCRIPTION~

Simple example processes some text one sentence at a time and adds them to a queue to be played, which shortens the wait time for audio to be played..

WARNING~

This script requires versions of 

INSTALLATION INSTRUCTIONS~

(1)  remember to create a virtual environment and activate it

(2)  pip install WhisperSpeech

WhisperSpeech relies on SpeechBrain, which automatically installs pytorch.  However, it reportedly installs a cpu-only version.  Thus, to use CUDA (which this script needs) you must run the following command after installing WhisperSpeech:

(3)  pip uninstall torch torchvision torchaudio

(4)  Then reinstall pytorch by going here to get the appropriate command:

https://pytorch.org/get-started/locally/

(5)  pip install soundfile # this might be redundant if WhisperSpeech makes soundfile part of its standard installation

(6)  pip install pydub

(7)  pip install sounddevice

(8) python text_to_audio_playback.py
'''

from pydub import AudioSegment
import numpy as np
import re
import threading
import queue
from whisperspeech.pipeline import Pipeline
import sounddevice as sd

# Initialize the pipeline with your desired model
# model_ref = 'collabora/whisperspeech:s2a-q4-small-en+pl.model'
# model_ref = 'collabora/whisperspeech:s2a-q4-tiny-en+pl.model'
model_ref = 'collabora/whisperspeech:s2a-q4-base-en+pl.model'

pipe = Pipeline(s2a_ref=model_ref)

# text to process
input_text = """
This script processes a body of text one sentence at a time and plays them consecutively.  This enables the audio playback to begin sooner instead of waiting for the entire body of text to be processed.  The script uses the threading and queue modules that are part of the standard Python library.  It also uses the sound device library, which is fairly reliable across different platforms.  I hope you enjoy, and feel free to modify or distribute at your pleasure.
"""

# Split into sentences for queue
sentences = re.split(r'[.!?;]+\s*', input_text)

# Queue to hold audio data
audio_queue = queue.Queue()

def process_text_to_audio(sentences, pipe):
    for sentence in sentences:
        if sentence:  # Ensure the sentence is not empty
            audio_tensor = pipe.generate(sentence)
            audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
            if len(audio_np.shape) == 1:
                audio_np = np.expand_dims(audio_np, axis=0)
            else:
                audio_np = audio_np.T
            audio_queue.put(audio_np)  # Put the audio numpy array into the queue
    audio_queue.put(None)  # Signal that processing is complete

def play_audio_from_queue(audio_queue):
    while True:
        audio_np = audio_queue.get()  # Wait for/get the next audio numpy array from the queue
        if audio_np is None:  # Check for the signal that processing is complete
            break
        try:
            sd.play(audio_np, samplerate=24000)
            sd.wait()  # Wait until audio is finished playing
        except Exception as e:
            print(f"Error playing audio: {e}")

# Create threads for processing and playback
processing_thread = threading.Thread(target=process_text_to_audio, args=(sentences, pipe))
playback_thread = threading.Thread(target=play_audio_from_queue, args=(audio_queue,))

# Start the threads
processing_thread.start()
playback_thread.start()

# Wait for both threads to complete
processing_thread.join()
playback_thread.join()