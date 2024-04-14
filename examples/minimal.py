from whisperspeech.pipeline import Pipeline

tts_pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model') # uncomment the line for the model you want to use
# tts_pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-base-en+pl.model') # uncomment the line for the model you want to use
# tts_pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model') # uncomment the line for the model you want to use

save_path = 'output.wav' # change the file extension to .mp3, .flac, .ogg etc. to save to a different file format
tts_pipe.generate_to_file(save_path, "This is a test") 
