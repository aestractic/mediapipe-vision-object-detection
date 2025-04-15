# mediapipe-vision-object-detection

This project or a VisionTasks implementation demonstrates object detection using Google's MediaPipe library with Python and OpenCV for images.

# mediapipe-image-detection

[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe Vision](https://img.shields.io/badge/MediaPipe%20Vision-brightgreen)](https://developers.google.com/mediapipe/solutions/vision)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?style=for-the-badge&logo=opencv&logoColor=black)](https://opencv.org/)

This repository contains Python code to perform object detection in images using Google's MediaPipe Vision library and OpenCV for image manipulation and visualization.

## Description

This project focuses on implementing object detection in static images using MediaPipe Vision's object detection models. The goal is to provide a clear and concise example of how to identify and locate multiple objects within image files.

## Key Features

* Clear implementation of object detection in images using MediaPipe.
* Utilization of OpenCV to load and visualize images with detections (bounding boxes and labels).
* Demonstrates the use of a pre-trained MediaPipe model (`efficientdet_lite_float32.tflite`).
* Includes options to configure the maximum number of detected objects and the confidence score threshold.
* Well-commented code for easy understanding.

## Requirements

Before running the code, ensure you have the following libraries installed:

* Python (version 3.6 or higher)
* VENV (To create a virtual environment for installing libraries without affecting your system)
* MediaPipe (`mediapipe`)
* OpenCV (`opencv-python`)

You can install the dependencies using pip:

Mediapipe already includes opencv

```bash
pip install mediapipe
```

After installation, you can see which libraries were installed in your environment (including the dependencies that came with MediaPipe) using the following command:

```bash
pip freeze
```
