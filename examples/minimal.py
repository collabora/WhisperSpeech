from whisperspeech.pipeline import Pipeline

tts_pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')

save_path = 'output.wav'
tts_pipe.generate_to_file(save_path, "This is a test") 