# README
This repository contains processing scripts for the MRSAudio dataset. 

### File Architecture
```
.
├── cut_drama_wav.py
├── cut_long.py
├── cut_dialogue_wav.py
├── cut.py
├── drama_align.py
├── euler_to_quaternion.py
├── gen_speed.py
├── merge_audio.py
├── mfa_preprocess.py
├── normalize.py
├── parse_drama.py
├── process_npy.py
├── README.md
├── resample.py
└── split_audio_2_10s.py
```

- The drama_align.py script is used to align text of Speech data and audio data.
- The drama_mfa_preprocess.py script is used to preprocess the drama audio files for MFA (Montreal Forced Aligner) alignment, which preprocess the text and generates the .lab files.
- The euler_to_quaternion.py script is used to convert the Euler angles to quaternion format.
- The gen_speed.py script is used to generate the speed of the moving sound source.
- The merge_audio.py script is used to merge the foa audio files and gopro video files.
- The mfa_preprocess.py script is used to preprocess the audio files for MFA (Montreal Forced Aligner) alignment, which preprocess the text and generates the .lab files.
- The normalize.py script is used to normalize the audio files to target loudness.
- The parse_drama.py script is used to parse the drama and match the text.
- The process_npy.py script is used to centralize the log file according to the ear position.
- The resample.py script is used to resample the audio files to 48 kHz.

### Aligning and Cutting Scripts 
- The cut_drama_wav.py script is used to cut the drama audio files into smaller segments according to the timestamps in the drama metadata.
- The cut_long.py script is used to cut the long audio segments into smaller segments less than 30 s.
- The cut_dialogue_wav.py script is used to cut the dialogue audio files into smaller segments according to the timestamps in the dialogue metadata.
- The split_audio_2_10s.py script is used to split the audio files into smaller segments, 10 s for default.
- The cut.py script is used to align all modality data, including audio, video, and motion data.


### Processing Steps
1. We align the audio and video files.
2. ASR is used to generate the text for dialogue audio files.
3. MFA is used to align the text and audio files.
4. We cut the long audio files into smaller segments. For speech, dialogue, music and singing, we cut the audio files according to the timestamps in the metadata. For sound files, we split the audio files into smaller segments, 10 s for default.
5. We generate prompts and annotates camera pose, sound source position, and speed for each audio segment.
6. We check the audio segments and corresponding annotations manually to ensure the quality of the dataset.
7. We centralize the log files according to the ear position.
8. We convert the Euler angles to quaternion format.