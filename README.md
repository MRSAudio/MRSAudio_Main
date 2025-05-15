# MRSAudio:  A Large-Scale Multimodal Recorded Spatial Audio Dataset with Refined Annotations

Humans rely on multisensory integration to perceive spatial environments, where auditory cues enable sound source localization in three-dimensional space. 
Despite the critical role of spatial audio in immersive technologies such as VR/AR, most existing multimodal datasets provide only monaural audio, which limits the development of spatial audio generation and understanding. 
To address these challenges, we introduce MRSAudio, a large-scale multimodal spatial audio dataset designed to advance research in spatial audio understanding and generation. 
MRSAudio spans four distinct components: MRSLife, MRSSpeech, MRSMusic, and MRSSing, covering diverse real-world scenarios. 
The dataset includes synchronized binaural and ambisonic audio, exocentric and egocentric video, motion trajectories, and fine-grained annotations such as transcripts, phoneme boundaries, lyrics, scores, and prompts.
To demonstrate the utility and versatility of MRSAudio, we establish five foundational tasks: audio spatialization, and spatial text to speech, spatial singing voice synthesis, spatial music generation and sound event localization and detection. 
Results show that MRSAudio enables high-quality spatial modeling and supports a broad range of spatial audio research.
Demos and dataset access are available at [MRSAudio](https://mrsaudio.github.io).

![image](head.png)

This repository contains the scripts for processing the MRSAudio dataset. The scripts are organized into different categories based on their functionality.

### File Architecture
```
.
├── compress_vid.py
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

- The compress_vid.py script compresses the video to 1920x1080 resolution with 24 fps.
- The cut_drama_wav.py script is used to cut the drama audio files into smaller segments according to the timestamps in the drama metadata.
- The cut_long.py script is used to cut the long audio segments into smaller segments less than 30 s.
- The cut_dialogue_wav.py script is used to cut the dialogue audio files into smaller segments according to the timestamps in the dialogue metadata.
- The cut.py script is used to align all modality data, including audio, video, and motion data.
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
- The split_audio_2_10s.py script is used to split the audio files into smaller segments, 10 s for default.

