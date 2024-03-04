'''
DESCRIPTION~

Enter text to be processed one sentence at a time, add them to a queue to be played, which shortens the wait time for audio to be played..

INSTALLATION INSTRUCTIONS~
**Tested on Windows

(1)  create a virtual environment and activate it
(2)  install pytorch by going to the following website and running the appropriate command for your platform and setup:

https://pytorch.org/get-started/locally/

(3)  pip3 install WhisperSpeech
(4)  pip3 install sounddevice==0.4.6
(5)  python gui_text_to_audio_playback.py
'''

from tkinter import *
import numpy as np
import re
import threading
import queue
from whisperspeech.pipeline import Pipeline
import sounddevice as sd

# uncomment the model you want to use
# model_ref = 'collabora/whisperspeech:s2a-q4-small-en+pl.model'
# model_ref = 'collabora/whisperspeech:s2a-q4-tiny-en+pl.model'
model_ref = 'collabora/whisperspeech:s2a-q4-base-en+pl.model'

pipe = Pipeline(s2a_ref=model_ref)

# Queue to hold audio data
audio_queue = queue.Queue()

def process_text_to_audio(sentences, pipe):
    for sentence in sentences:  # Iterate through each sentence
        if sentence:  # Ensure the sentence is not empty
            audio_tensor = pipe.generate(sentence)  # Generate audio tensor for the sentence
            audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)  # Convert tensor to numpy array, scale, and cast to int16
            if len(audio_np.shape) == 1:  # Check if the numpy array is 1D
                audio_np = np.expand_dims(audio_np, axis=0)  # Add a new dimension to make it 2D
            else:
                audio_np = audio_np.T  # Transpose the numpy array if it's not 1D
            audio_queue.put(audio_np)  # Put the audio numpy array into the queue
    audio_queue.put(None)  # Signal that processing is complete

def play_audio_from_queue(audio_queue):
    while True:
        audio_np = audio_queue.get()
        if audio_np is None:
            break
        try:
            sd.play(audio_np, samplerate=24000)
            sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")

def start_processing():
    user_input = text_input.get("1.0", "end-1c")
    sentences = re.split(r'[.!?;]+\s*', user_input)
    
    while not audio_queue.empty():
        audio_queue.get()
    
    processing_thread = threading.Thread(target=process_text_to_audio, args=(sentences, pipe))
    playback_thread = threading.Thread(target=play_audio_from_queue, args=(audio_queue,))
    
    processing_thread.start()
    playback_thread.start()

root = Tk()
root.title("Text to Speech")

text_input = Text(root, height=10, width=50)
text_input.pack()

process_button = Button(root, text="Text to Speech", command=start_processing)
process_button.pack()

root.mainloop()
