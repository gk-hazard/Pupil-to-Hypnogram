# Pupil-based hypnogram
This repository contains Python and MATLAB codes for our paper ([Keio J Med 2023](https://doi.org/10.2302/kjm.2022-0020-OA) and [bioRxiv](https://biorxiv.org/cgi/content/short/2022.08.06.503067v1)) __"Pupil dynamics-derived sleep stage classification of a head-fixed mouse using a recurrent neural network"__. Sample data are available at [Mendeley Data](https://doi.org/10.17632/rr4gc6mybg.1).

![P2Hgithub](https://user-images.githubusercontent.com/78021878/183251903-d4405d1e-f726-40ab-9bb6-3092c67f6ce2.gif)

## Highlights
-	A pupil dynamics-based vigilance state classification of a head-fixed mouse using a log short-term memory (LSTM) model was proposed.
-   The necessary inputs are 1) a pupil video in MPEG-4 format, and 2) its timestamp in a text format.
-   Vigilance states (REM, NREM, WAKE) are estimated every 10 s.
-   The LSTM model achieved an accuracy of 86%, and a macro averaged F1-score of 0.77.
-	A pupil-based hypnogram would be particularly compatible with wide-field imaging of cortical activity.

## Workfolw
![githubworkflow](https://user-images.githubusercontent.com/78021878/184583324-f9242d74-7874-438d-90a4-c920c0a70a6f.png)

## Steps for a pupil-based hypnogram
### 0. Preparation of a pupil movie of a head-fixed mouse
Prior habituation of a mouse for a head restrained condition is necesasry.

We used a conventional USB camera (BSW200MBK; Buffalo Inc.) and an IR light (940 nm, FRS5 JS; OptoSupply Ltd.).
- A pupil video in MPEG-4 format
- Its timestamp in a text format

### 1. Pupil tracking with DeepLabCut (DLC)
We explain the steps using the *file names* of the sample data at [Mendeley Data](https://doi.org/10.17632/rr4gc6mybg.1) (Put the sample data [Inputs, Intermediate, and Outputs folders] directly in the Data folder).
- Code:
    - `DLC.ipynb`
- Input: 
    - A pupil video (*SNT267_0806.mp4*)
- Outputs: 
    - A csv file for time-series of pupil position coordinate (*SNT267_0806DLC_resnet50_DLCJul19shuffle1_1030000.csv*)
    - A labeled pupil video (*SNT267_0806DLC_resnet50_DLCJul19shuffle1_280000_labeled.mp4*)
    
### 2. Pupil feature extraction with a MATLAB code
- Code:
    - `GetPupilfeature.mlx`
- Inputs:
    - The csv file for pupil coordinates
    - The timestamp file (*video_01.txt*)
- Outputs:
    - Pupil feature csv files
        - Pupil Diameter (*PD_ratio10Hz.csv*)
        - Pupil center Location (*PupilLocation10Hz.csv*)
        - Pupil Velocity (*Pupil_velocity10Hz.csv*)
        - Eyelid Opening (*EyeOpening_ratio10Hz.csv*)

### 3. Vigilance states estimation with an LSTM model
- Code:
    - `LSTM.ipynb`
- Inputs:
    - The pupil feature csv files
- Outputs:
    - A csv file for a pupil-derived hypnogram (*estimatedHypnoScore.csv*)
    - A figure for a pupil-derived hypnogram (*LSTMestimation.png*)

## About a timestamp of an mp4 file
While you might be able to retrieve timestamps of a mp4 file using the code below, they might be inaccurate. We used a custom written LabVIEW code to obtain timestamps in our study.
- Code:
    - `GetTextFrommp4.ipynb`
- Input: 
    - a pupil video (*SNT267_0806.mp4*)
- Output:
    - a timestamp text file (*video_02.txt*)

## Software environment
- DeepLabCut ver. 2.2rc3
- Matlab 2020a
- PyTorch ver. 1.12.0
