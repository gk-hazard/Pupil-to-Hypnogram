# Codes for the manuscript: Pupil dynamics-derived sleep stage classification of a head-fixed mouse using a recurrent neural network
![P2Hgithub](https://user-images.githubusercontent.com/78021878/183251903-d4405d1e-f726-40ab-9bb6-3092c67f6ce2.gif)



This repository contains MATLAB and Python code for our papers ([Keio Journal of Medicine]() and [bioRxiv](https://biorxiv.org/cgi/content/short/2022.08.06.503067v1))__"Pupil dynamics-derived sleep stage classification of a head-fixed mouse using a recurrent neural network"__

an End to end sleep stage classification using pupil dynamics is available.

## Highlights
-	A pupil dynamics-based vigilance state classification of a head-fixes mouse using a log short-term memory (LSTM) model was proposed.
-	The LSTM model achieved high classification performance (macro F1 score, 0.77; accuracy, 86%) using 10 s pupil dynamics as input.
-	Our method using pupil dynamics would be compatible to a wide-field calcium imaging and functional MRI of the cortex.
-	Sample data for implementing vigilance state classification from pupil dynamics is available at https://data.mendeley.com/datasets/rr4gc6mybg/1

## Steps to create hypnograms from pupil videos
### 1. DeepLabCut (DLC)
### 2. MATLAB (extract feature inputs from pupil time-series coordinates)
### 3. LSTM

## Software environment
