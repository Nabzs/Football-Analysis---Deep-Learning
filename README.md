# Football Analysis - Deep Learning

## Introduction
This project leverages deep learning and artificial intelligence to automatically detect and track players, referees, and the football in match videos. Trained using Google Colab with the YOLO model, it also assigns players to teams via color-based clustering, estimates ball possession, accounts for camera movement, and calculates player speed and distance covered using advanced computer vision techniques.

![Screenshot](output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](models/best.pt)
- [Trained Yolo v5](models/last.pt)


## Sample video
-  [Sample input video](input_videos/08fd33_4.mp4)

## Output video
-  [Final result](output_videos/output_video.avi)

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas