# Facial Video-Based Heart Rate Estimation

### DeepPhys-Based BPM Estimation from Facial Video Using rPPG

## Overview

This project focuses on estimating heart rate in beats per minute (BPM) from facial video using remote photoplethysmography (rPPG).

The pipeline uses DeepPhys to extract physiological signals from facial video frames. Haar Cascade and MTCNN are used for face detection and ROI extraction before the frames are processed by the model.

## Problem

Heart rate estimation from facial video is challenging because physiological signals are subtle and can be affected by facial movement, lighting conditions, motion blur, and background noise.

The objective is to estimate BPM from facial video without requiring a contact-based heart rate sensor.

## Approach

The overall pipeline is:

Facial Video → Face Detection → ROI Extraction → Motion + Appearance → DeepPhys → FFT → BPM Estimation

### Face Detection and ROI Extraction

Two face detection approaches are implemented:

* Haar Cascade
* MTCNN

The detected face region is extracted as the ROI with additional padding before being passed to the preprocessing pipeline.

### DeepPhys Processing

The extracted facial frames are processed to generate:

* Appearance frames
* Motion frames using frame-to-frame differences
* Combined motion-appearance inputs

These inputs are passed to the DeepPhys model to estimate the underlying physiological signal.

### Heart Rate Estimation

The predicted physiological signal is processed using Fast Fourier Transform (FFT) to identify the dominant frequency component.

The dominant frequency is converted to BPM:

BPM = Frequency (Hz) × 60

A smoothing step is also applied to reduce variations between consecutive video chunks.

## Model Performance

The pipeline was evaluated using a 60-second test video, processed in 5-second chunks.

The total processing time was:

* Haar Cascade: 7.14 seconds
* MTCNN: 6.95 seconds

Both implementations processed the video faster than real time, demonstrating efficient processing performance.

Some variation in BPM estimates was observed between chunks due to facial movement and changes in signal quality, which are common challenges in rPPG systems.

## Failure Cases

The pipeline may produce less reliable BPM estimates under challenging conditions, including:

* Significant head movement or motion blur
* Partial face occlusion
* Low lighting conditions
* Poor video quality
* Variations in facial signal quality

During video preprocessing, an incomplete segment could occasionally be generated while splitting the video into 5-second chunks. Invalid or incomplete chunks were filtered before model inference to ensure consistent segment processing.

## Technologies

* Python
* PyTorch
* OpenCV
* NumPy
* MTCNN
* Haar Cascade
* DeepPhys

## Project Structure

```text
Facial-Video-Heart-Rate-Estimation/
│
├── Model/
│   └── DeepPhys.py
│
├── src/
│   ├── haarcascade_heart_rate.py
│   └── mtcnn_heart_rate.py
│
├── README.md
├── requirements.txt
└── .gitignore
```
