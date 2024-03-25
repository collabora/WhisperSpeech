# Benchmark:

### Summary
WhisperSpeech contains two primary models s2a and t2s (plus a vocoder) when covnerting text to speech.  You can specify which size/type of model to use in a python script.  The below data is from testing all available model combinations as of 3/24/2024, except for four (4) t2s models that were unable to be tested.  More details are contained in the docstring for the benchmarking script mentioned below.

### Testing Procedure
- Hardware: RTX 4090, 13900k cpu
- Software: Windows 10, CUDA 12.3, torch==2.2.0, Whisper Speech (source code from repo as of 3/24/2024).
- Parameters: Neither Flash Attention 2 nor `torch.compile` were used, which provide significant speedups if you can run them.
- Testing Procedure: Each permutation of model combinations (S2A and T2S) were run three times. The numbers below represent the average from those three runs.  The test itself consisted of converting the following passage to speech:

<details>
  <summary>PASSAGE</summary>
  "This script processes a body of text one sentence at a time and plays them consecutively. This enables the audio playback to begin sooner instead of waiting for the entire body of text to be processed. The script uses the threading and queue modules that are part of the standard Python library. It also uses the sound device library, which is fairly reliable across different platforms. I hope you enjoy, and feel free to modify or distribute at your pleasure."
</details>

- The "Average First Segment Processing Time" refers to the time to process the first sentence.  This is important because most scripts should begin playback after processing the first segment while the others continue to be processed, resulting in near-real time response.
- The test was done using the `benchmark_whisperspeech.py` script located in the "Examples" folder. See the detailed docstring within that script for more details regarding how the data was accurately collected.
- Contributed by [BBC-Esq](https://github.com/BBC-Esq)


<table>
  <thead>
    <tr>
      <th>S2A Model</th>
      <th>T2S Model</th>
      <th>Average Total Processing Time (s)</th>
      <th>Average First Segment Processing Time (s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>tiny-en+pl</td>
      <td>tiny-en+pl</td>
      <td align="center">15.06</td>
      <td align="center">2.90</td>
    </tr>
    <tr>
      <td>hq-fast-en+pl</td>
      <td>tiny-en+pl</td>
      <td align="center">15.17</td>
      <td align="center">2.96</td>
    </tr>
    <tr>
      <td>hq-fast-en+pl</td>
      <td>hq-fast-en+pl</td>
      <td align="center">15.29</td>
      <td align="center">2.98</td>
    </tr>
    <tr>
      <td>tiny-en+pl</td>
      <td>hq-fast-en+pl</td>
      <td align="center">15.29</td>
      <td align="center">2.99</td>
    </tr>
    <tr>
      <td>tiny-en+pl</td>
      <td>base-en+pl</td>
      <td align="center">16.09</td>
      <td align="center">3.13</td>
    </tr>
    <tr>
      <td>hq-fast-en+pl</td>
      <td>base-en+pl</td>
      <td align="center">16.35</td>
      <td align="center">3.16</td>
    </tr>
    <tr>
      <td>tiny-en+pl</td>
      <td>fast-medium-en+pl+yt</td>
      <td align="center">17.38</td>
      <td align="center">3.36</td>
    </tr>
    <tr>
      <td>hq-fast-en+pl</td>
      <td>fast-medium-en+pl+yt</td>
      <td align="center">17.50</td>
      <td align="center">3.40</td>
    </tr>
    <tr>
      <td>tiny-en+pl</td>
      <td>small-en+pl</td>
      <td align="center">19.66</td>
      <td align="center">3.83</td>
    </tr>
    <tr>
      <td>hq-fast-en+pl</td>
      <td>small-en+pl</td>
      <td align="center">20.05</td>
      <td align="center">3.87</td>
    </tr>
    <tr>
      <td>base-en+pl</td>
      <td>tiny-en+pl</td>
      <td align="center">20.76</td>
      <td align="center">4.06</td>
    </tr>
    <tr>
      <td>base-en+pl</td>
      <td>hq-fast-en+pl</td>
      <td align="center">21.26</td>
      <td align="center">4.21</td>
    </tr>
    <tr>
      <td>base-en+pl</td>
      <td>base-en+pl</td>
      <td align="center">21.94</td>
      <td align="center">4.31</td>
    </tr>
    <tr>
      <td>base-en+pl</td>
      <td>fast-medium-en+pl+yt</td>
      <td align="center">23.32</td>
      <td align="center">4.43</td>
    </tr>
    <tr>
      <td>base-en+pl</td>
      <td>small-en+pl</td>
      <td align="center">25.31</td>
      <td align="center">4.91</td>
    </tr>
    <tr>
      <td>small-en+pl</td>
      <td>tiny-en+pl</td>
      <td align="center">37.34</td>
      <td align="center">7.31</td>
    </tr>
    <tr>
      <td>small-en+pl</td>
      <td>hq-fast-en+pl</td>
      <td align="center">37.65</td>
      <td align="center">7.40</td>
    </tr>
    <tr>
      <td>small-en+pl</td>
      <td>base-en+pl</td>
      <td align="center">38.78</td>
      <td align="center">7.49</td>
    </tr>
    <tr>
      <td>small-en+pl</td>
      <td>fast-medium-en+pl+yt</td>
      <td align="center">39.59</td>
      <td align="center">7.68</td>
    </tr>
    <tr>
      <td>small-en+pl</td>
      <td>small-en+pl</td>
      <td align="center">42.05</td>
      <td align="center">8.11</td>
    </tr>
  </tbody>
</table>
