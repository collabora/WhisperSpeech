'''
DESCRIPTION~

Type text or load a file to be converted TTS, sentence by sentence, to quickstart playback.

INSTALLATION INSTRUCTIONS~
**Tested on Windows

(1)  create a virtual environment and activate it
(2)  install pytorch by going to the following website and running the appropriate command for your platform and setup:

https://pytorch.org/get-started/locally/

(3)  run pip3 install WhisperSpeech

(4)  pip3 install sounddevice==0.4.6 pypdf==4.0.2 python-docx==1.1.0 nltk==3.8.1

(9)  python gui_file_to_text_to_audio_playback.py
'''

from tkinter import *
from tkinter import filedialog
import numpy as np
import re
import threading
import queue
from whisperspeech.pipeline import Pipeline
import sounddevice as sd
from pypdf import PdfReader
from docx import Document
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

main_bg = "#1B2A2F"
widget_bg = "#303F46"
button_bg = "#263238"
label_bg = "#1E272C"
text_fg = "#FFFFFF"

# uncomment the model you want to use
# model_ref = 'collabora/whisperspeech:s2a-q4-small-en+pl.model'
# model_ref = 'collabora/whisperspeech:s2a-q4-tiny-en+pl.model'
model_ref = 'collabora/whisperspeech:s2a-q4-base-en+pl.model'

pipe = Pipeline(s2a_ref=model_ref)

# Queue to hold audio data
audio_queue = queue.Queue()

class TextUtilities:
    def pdf_to_text(self, pdf_file_path):
        reader = PdfReader(pdf_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ''
        return text

    def docx_to_text(self, docx_file_path):
        doc = Document(docx_file_path)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])

    def txt_to_text(self, txt_file_path):
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def clean_text(self, text):
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        text = text.replace('\n', ' ')
        sentences = sent_tokenize(text)
        cleaned_text = ' '.join(sentences)
        return cleaned_text

def process_text_to_audio(sentences, pipe):
    for sentence in sentences:  # Iterate through each sentence in the list
        if sentence:  # Check if sentence is not empty
            audio_tensor = pipe.generate(sentence)  # Generate audio tensor for the sentence using the pipeline
            audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)  # Convert tensor to numpy array, scale, and typecast
            if len(audio_np.shape) == 1:  # Check if the numpy array is 1D
                audio_np = np.expand_dims(audio_np, axis=0)  # Make the numpy array 2D for compatibility
            else:
                audio_np = audio_np.T  # Transpose the array to match expected input shape
            audio_queue.put(audio_np)  # Place the audio numpy array into the queue
    audio_queue.put(None)  # Signal that processing is complete by putting None

def play_audio_from_queue(audio_queue):
    while True:  # Continuously attempt to play audio from the queue
        audio_np = audio_queue.get()  # Get the next numpy array (audio data) from the queue
        if audio_np is None:  # Check if the signal to stop (None) is received
            break  # Exit the loop if None is received
        try:
            sd.play(audio_np, samplerate=24000)  # Attempt to play the audio data
            sd.wait()  # Wait for the playback to finish
        except Exception as e:
            print(f"Error playing audio: {e}")  # Print any errors during playback

def start_processing():
    user_input = text_input.get("1.0", "end-1c")  # Get text from the text input widget
    sentences = sent_tokenize(user_input)  # Use NLTK's sent_tokenize to define a sentence
    while not audio_queue.empty():  # Ensure the audio queue is empty before starting
        audio_queue.get()  # Remove and discard any remaining items in the queue
    processing_thread = threading.Thread(target=process_text_to_audio, args=(sentences, pipe))  # Set up thread for text-to-audio processing
    playback_thread = threading.Thread(target=play_audio_from_queue, args=(audio_queue,))  # Set up thread for audio playback
    processing_thread.start()  # Start the processing thread
    playback_thread.start()  # Start the playback thread

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        text = process_file(file_path)
        text_input.delete("1.0", END)
        text_input.insert("1.0", text)

def process_file(file_path):
    utilities = TextUtilities()
    text = ""
    if file_path.lower().endswith('.pdf'):
        text = utilities.pdf_to_text(file_path)
    elif file_path.lower().endswith('.docx'):
        text = utilities.docx_to_text(file_path)
    elif file_path.lower().endswith(('.txt', '.py', '.html', '.md')):
        text = utilities.txt_to_text(file_path)
    text = utilities.clean_text(text)
    return text

root = Tk()
root.title("Text to Speech")

# Make the window always on top
root.attributes('-topmost', 1)

root.configure(bg=main_bg)

root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

text_input = Text(root, bg=widget_bg, fg=text_fg)
text_input.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

process_button = Button(root, text="Text to Speech", command=start_processing, bg=button_bg, fg=text_fg)
process_button.grid(row=1, column=0, sticky='ew', padx=10)

file_button = Button(root, text="Extract Text from File", command=select_file, bg=button_bg, fg=text_fg)
file_button.grid(row=2, column=0, sticky='ew', padx=10)

support_label = Label(root, text="Supports .pdf (with OCR already done on them), .docx, and .txt files.", bg=label_bg, fg=text_fg)
support_label.grid(row=3, column=0, sticky='ew')

root.mainloop()
