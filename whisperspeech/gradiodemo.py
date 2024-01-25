import tempfile
import gradio as gr
import os
import re
import torch
import soundfile as sf
import numpy as np
from whisperspeech.pipeline import Pipeline
from whisperspeech.languages import LANGUAGES
from whisperspeech.utils import resampler

pipe = Pipeline()
if torch.cuda.is_available():
    pipe.to('cuda')

title = """# 🙋🏻‍♂️ Welcome to🌟Collabora🌬️💬📝WhisperSpeech
You can use this ZeroGPU Space to test out the current model [🌬️💬📝collabora/whisperspeech](https://huggingface.co/collabora/whisperspeech). 🌬️💬📝collabora/whisperspeech is An Open Source text-to-speech system built by inverting Whisper. Install it and use your command line interface locally with `pip install whisperspeech`. It's like Stable Diffusion but for speech – both powerful and easily customizable : so you can use it programmatically in your own pipelines! [Contribute to whisperspeech here](https://github.com/collabora/WhisperSpeech) 
You can also use 🌬️💬📝WhisperSpeech by cloning this space. 🧬🔬🔍 Simply click here: <a style="display:inline-block" href="https://huggingface.co/spaces/Tonic/laion-whisper?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></h3> 
We're **celebrating the release of the whisperspeech** at [the LAION community, if you love open source ai learn more here : https://laion.ai/](https://laion.ai/) big thanks to the folks at huggingface for the community grant 🤗
### How to Use
Input text with the language identifiers provided to create a multilingual speech. Optionally you can add an audiosample to make a voice print."""

def parse_multilingual_text(input_text):
    pattern = r"<(\w+)>\s(.*?)\s(?=<\w+>|$)"
    segments = re.findall(pattern, input_text)
    return [(lang, text.strip()) for lang, text in segments if lang in LANGUAGES.keys()]

def generate_segment_audio(text, lang, speaker_url):
    if not isinstance(text, str):
        text = text.decode("utf-8") if isinstance(text, bytes) else str(text)

    audio_data = pipe.generate(text, speaker_url, lang)
    resample_audio = resampler(newsr=24000)
    audio_data_resampled = next(resample_audio([{'sample_rate': 24000, 'samples': audio_data.cpu()}]))['samples_24k']
    audio_np = audio_data_resampled.cpu().numpy()
    return audio_np

def concatenate_audio_segments(segments):
    concatenated_audio = np.concatenate(segments, axis=1)
    return concatenated_audio

def whisper_speech_demo(multilingual_text, speaker_audio):
    segments = parse_multilingual_text(multilingual_text)
    if not segments:
        return None, "No valid language segments found. Please use the format: <lang> text"

    speaker_url = speaker_audio if speaker_audio is not None else None
    audio_segments = [generate_segment_audio(text, lang, speaker_url) for lang, text in segments]
    concatenated_audio = concatenate_audio_segments(audio_segments)
    concatenated_audio = concatenated_audio / np.max(np.abs(concatenated_audio))

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, concatenated_audio.T, 24000, format='WAV', subtype='PCM_16')
        return tmp_file.name

with gr.Blocks() as demo:
    gr.Markdown(title)
    output_audio = gr.Audio(label="Generated Speech")
    generate_button = gr.Button("Try 🌟Collabora🌬️💬📝WhisperSpeech")
    with gr.Row():
        text_input = gr.Textbox(label="Enter multilingual text", placeholder="e.g., <en> Hello <fr> Bonjour <es> Hola")
        speaker_input = gr.Audio(label="Upload or Record Speaker Audio (optional)", sources=["upload", "microphone"])
        with gr.Accordion("Available Languages and Their Tags"):
            language_list = "\n".join([f"{lang}: {LANGUAGES[lang]}" for lang in LANGUAGES])
            gr.Markdown(language_list)    
    generate_button.click(whisper_speech_demo, inputs=[text_input, speaker_input], outputs=output_audio)

demo.launch()
