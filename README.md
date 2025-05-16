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

This repository contains the scripts for data processing and evaluation metrics of MRSAudio dataset. The scripts are organized into different categories based on their functionality.

### File Architecture
```
.
├── Audio_Spatialization
├── audio-visual-seld-dcase2023_
├── data_process
├── head.png
├── README.md
├── MAA-2
├── rmssinger
├── spatial_eval
└── spatial_fad
```
