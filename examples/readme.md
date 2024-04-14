# Example Scripts

Contributions are welcome! Feel free to create an issue or pull request on GitHub.

### `minimal.py`

- Minimalistic script that takes hardcoded text input and outputs an audio file.

### `text_to_playback.py`

- Utilizes the new `generate_to_playback` method to directly convert hardcoded text to audio playback without intermediate steps. Designed for minimal script length, but does not include queue management to reduce latency.

### `text_to_audio_playback.py`

- Processes text one sentence at a time and adds them to a queue for playback. Designed for users who prefer a command-line approach but still want the efficiency of queued playback.

### `gui_file_to_text_to_audio_playback.py`

- Provides a graphical user interface allowing users to load a file. The text is then converted into speech, sentence by sentence using queue management in order to reduce latency.

### `gui_text_to_audio_playback.py`

- Similar to `gui_file_to_text_to_audio_playback.py`, but a user simply enters the text to be played back.  Text is still processed one sentence at a time for low latency.


| Feature                           | gui_file_to...<br>audio_playback.py | gui_text_to...<br>audio_playback.py | minimal.py | text_to_audio...<br>playback.py | text_to_playback.py |
|:---------------------------------:|:-----------------------------------:|:-----------------------------------:|:----------:|:-------------------------------:|:-------------------:|
| **GUI**                   | <center>✅</center>                  | <center>✅</center>                  | <center>❌</center> | <center>❌</center>             | <center>❌</center>  |
| **Input**                  | File                                | Text Entry                          | Predefined Text | Predefined Text                 | Predefined Text     |
| **Output**                 | Audio Playback                      | Audio Playback                      | WAV File   | Audio Playback                  | Audio Playback      |
| **Queue Management** | <center>✅</center>                  | <center>✅</center>                  | <center>❌</center> | <center>✅</center>            | <center>❌</center>  |
| **Text-to-Speech<br> Conversion**| <center>✅</center>                  | <center>✅</center>                  | <center>✅</center> | <center>✅</center>            | <center>✅</center>  |
| **Load File**              | <center>✅</center>                  | <center>❌</center>                  | <center>❌</center> | <center>❌</center>             | <center>❌</center>  |
