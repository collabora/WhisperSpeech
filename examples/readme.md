# Example Scripts

Contributions are welcome! If you have suggestions for improvements or new features, feel free to create an issue or pull request on GitHub.

### `gui_file_to_text_to_audio_playback.py`

- **Description**: This script provides a graphical user interface allowing users to load a file. The text from the file is then converted into speech, sentence by sentence, to enable quickstart audio playback.

### `gui_text_to_audio_playback.py`

- **Description**: Similar to `gui_file_to_text_to_audio_playback.py`, this script offers a GUI for users to enter text manually. The text is processed one sentence at a time, and sentences are added to a queue for playback. This method reduces the waiting time for audio playback, making it efficient for interactive use.

### `minimal.py`

- **Description**: As the name suggests, this is a minimalistic script that takes text input and outputs an audio file. It's streamlined for simplicity and quick use, without the additional features of queue management or GUI.

### `text_to_audio_file.py`

- **Description**: This simple script demonstrates the basic functionality of converting text to speech. It speaks the provided text and creates an audio file named `output_audio.wav` in the script's directory.

### `text_to_audio_playback.py`

- **Description**: This script processes text one sentence at a time and adds them to a queue for playback, similar to `gui_text_to_audio_playback.py` but without the graphical interface. It's designed for users who prefer a command-line approach but still want the efficiency of queued playback.

| Feature                           | gui_file_to...<br>audio_playback.py | gui_text_to...<br>audio_playback.py | minimal.py | text_to_audio...<br>file.py | text_to_audio...<br>playback.py |
|:---------------------------------:|:-----------------------------------:|:-----------------------------------:|:----------:|:-------------------------:|:-------------------------------:|
| **GUI Support**                   | <center>✅</center>                  | <center>✅</center>                  | <center>❌</center> | <center>❌</center>       | <center>❌</center>             |
| **Input Method**                  | File                                | Text Entry                          | Text Entry | Text Entry                | Text Entry                      |
| **Output Format**                 | Audio Playback                      | Audio Playback                      | WAV File   | WAV File                  | Audio Playback                  |
| **Queue Management<br>for Playback** | <center>✅</center>                  | <center>✅</center>                  | <center>❌</center> | <center>❌</center>       | <center>✅</center>            |
| **Live Text-to-Speech<br> Conversion**| <center>✅</center>                  | <center>✅</center>                  | <center>✅</center> | <center>✅</center>       | <center>✅</center>            |
| **Batch Processing**              | <center>✅</center>                  | <center>❌</center>                  | <center>❌</center> | <center>❌</center>       | <center>❌</center>             |
