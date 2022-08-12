# Codes for the manuscript: Pupil dynamics-derived sleep stage classification of a head-fixed mouse using a recurrent neural network
![P2Hgithub](https://user-images.githubusercontent.com/78021878/183251903-d4405d1e-f726-40ab-9bb6-3092c67f6ce2.gif)



This repository contains MATLAB and Python code for our papers ([bioRxiv](https://biorxiv.org/cgi/content/short/2022.08.06.503067v1))__"Pupil dynamics-derived sleep stage classification of a head-fixed mouse using a recurrent neural network"__

an End to end sleep stage classification using pupil dynamics is available.

## Workfolw
Only a **pupil video** and its **timestamp** are required to conduct vigilance state classification.
**you can get timestamp from mp4 even if you don't have it.**
![githubworkflow](https://user-images.githubusercontent.com/78021878/183596775-50cd8868-1985-40a0-baff-8ff83e67b2c1.png)

## Highlights
-	A pupil dynamics-based vigilance state classification of a head-fixes mouse using a log short-term memory (LSTM) model was proposed.
-	The LSTM model achieved high classification performance (macro F1 score, 0.77; accuracy, 86%) using 10 s pupil dynamics as input.
-	Our method using pupil dynamics would be compatible to a wide-field calcium imaging and functional MRI of the cortex.
-	Sample data for implementing vigilance state classification from pupil dynamics is available at https://data.mendeley.com/datasets/rr4gc6mybg/1

## Steps to create hypnograms from pupil videos
### 1. DeepLabCut (DLC)
Obtain pupil coordinates from a left pupil video
- Inputs: 
    - SNT267_0806.mp4 (pupil video)

- Outputs: 
    - SNT267_0806DLC_resnet50_DLCJul19shuffle1_280000_labeled.mp4 (a labeled pupil video)
    - SNT267_0806DLC_resnet50_DLCJul19shuffle1_1030000.csv (coordinates which characterizes the pupil)
### 2. MATLAB
Extract feature inputs from pupil time-series coordinates.
-   Inputs:
    -   SNT267_0806DLC_resnet50_DLCJul19shuffle1_1030000.csv
    - video_01.txt (video timestamp)
- Outputs:
    - PD_ratio10Hz.csv (Pupil Diameter)
    - PupilLocation10Hz.csv (Pupil center Location)
    - Pupil_velocity10Hz.csv (Pupil Velocity)
    - EyeOpening_ratio10Hz.csv (Eyelid Opeining)

### 3. LSTM
Estimate vigilance states from pupil dynamics features.
- Inputs:
    - PD_ratio10Hz.csv
    - PupilLocation10Hz.csv
    - Pupil_velocity10Hz.csv
    - EyeOpening_ratio10Hz.csv
- Outputs:
    - estimatedHypnoScore.csv (HypnoScore)
    - LSTMestimation.png

## Way to get timestamp from mp4
- Inputs: 
    - SNT267_0806.mp4 (pupil video)
- Outputs:
    - video_02.txt
## Software environment
