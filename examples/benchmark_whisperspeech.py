import numpy as np
import re
import queue
from whisperspeech.pipeline import Pipeline
import sounddevice as sd
import time
import torch
import gc

"""
This script benchmarks the performance of different combinations of speech-to-text (S2A) and text-to-speech (T2S) models
from the WhisperSpeech library. It processes a given input text by splitting it into sentences and generating audio
for each sentence using the selected S2A and T2S models.

The core functionality revolves around generating audio for each sentence independently, allowing audio playback to commence without the need to wait for the entire text block's processing. This is achieved through a combination of threading and queuing, ensuring efficient task distribution and execution.

The script iterates over all combinations of the specified S2A and T2S models, running each combination three times
to calculate the average total processing time and the average first segment processing time.

The commented out t2s models below were not tested.

Note: The script assumes that the required dependencies are installed and properly configured.
"""

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

s2a_models = [
    'collabora/whisperspeech:s2a-q4-base-en+pl.model',
    'collabora/whisperspeech:s2a-q4-hq-fast-en+pl.model',
    'collabora/whisperspeech:s2a-q4-small-en+pl.model',
    'collabora/whisperspeech:s2a-q4-tiny-en+pl.model'
]

t2s_models = [
    'collabora/whisperspeech:t2s-base-en+pl.model',
    'collabora/whisperspeech:t2s-fast-medium-en+pl+yt.model',
    #'collabora/whisperspeech:t2s-fast-small-en+pl.model',
    #'collabora/whisperspeech:t2s-fast-small-en+pl+yt.model',
    #'collabora/whisperspeech:t2s-fast-small-nocps-en+pl+yt.model',
    'collabora/whisperspeech:t2s-hq-fast-en+pl.model',
    'collabora/whisperspeech:t2s-small-en+pl.model',
    #'collabora/whisperspeech:t2s-small-yt.model',
    'collabora/whisperspeech:t2s-tiny-en+pl.model'
]

input_text = """
This script processes a body of text one sentence at a time and plays them consecutively. This enables the audio playback to begin sooner instead of waiting for the entire body of text to be processed. The script uses the threading and queue modules that are part of the standard Python library. It also uses the sound device library, which is fairly reliable across different platforms. I hope you enjoy, and feel free to modify or distribute at your pleasure.
"""
sentences = re.split(r'[.!?;]+\s*', input_text.strip())

benchmark_results = {}
first_segment_times = {}

for s2a_ref in s2a_models:
    for t2s_ref in t2s_models:
        total_processing_times = []
        first_segment_processing_times = []
        
        for _ in range(3):
            cleanup()
            
            pipe = Pipeline(s2a_ref=s2a_ref, t2s_ref=t2s_ref)
            
            total_start_time = time.time()

            for i, sentence in enumerate(sentences):
                if sentence:
                    if i == 0:
                        first_segment_start_time = time.time()
                    
                    audio_tensor = pipe.generate(sentence)
                    audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
                    if len(audio_np.shape) == 1:
                        audio_np = np.expand_dims(audio_np, axis=0)
                    else:
                        audio_np = audio_np.T
                    
                    if i == 0:
                        first_segment_end_time = time.time()
                        first_segment_time = first_segment_end_time - first_segment_start_time
                        first_segment_processing_times.append(first_segment_time)

            total_end_time = time.time()
            total_processing_time = total_end_time - total_start_time
            total_processing_times.append(total_processing_time)
            
            time.sleep(2)
        
        avg_total_processing_time = sum(total_processing_times) / len(total_processing_times)
        avg_first_segment_processing_time = sum(first_segment_processing_times) / len(first_segment_processing_times)
        
        benchmark_results[(s2a_ref, t2s_ref)] = avg_total_processing_time
        first_segment_times[(s2a_ref, t2s_ref)] = avg_first_segment_processing_time

for models in benchmark_results:
    avg_total_time = benchmark_results[models]
    avg_first_segment_time = first_segment_times[models]
    print(f"S2A Model: {models[0]}, T2S Model: {models[1]}, Average Total Processing Time: {avg_total_time:.2f} seconds, Average First Segment Processing Time: {avg_first_segment_time:.2f} seconds")