# Real-Time Person Background Removal

Real-Time Person Background Removal is a Python project designed for removing backgrounds from live video feeds, focusing on accurate segmentation of individuals using YOLO (You Only Look Once) models and OpenCV. This project provides a seamless solution for applications requiring background-free video streams, such as video conferencing, virtual backgrounds, and augmented reality.

## Features

- **Precise Segmentation**: Utilizes YOLO models optimized for precise person detection and segmentation.
- **Real-Time Processing**: Integrates OpenCV for smooth real-time video processing, ensuring minimal latency during live video streams.
- **Model Selection**: Allows users to choose from a variety of YOLO models to suit different requirements and hardware capabilities.
- **ONNX Support**: Supports ONNX (Open Neural Network Exchange) format for enhanced model deployment and interoperability.

## Usage

1. **Environment Setup**: Ensure Python 3.x is installed on your system.
2. **Dependencies Installation**: Install the necessary dependencies using `pip install -r requirements.txt`.
3. **Model Selection**: Run the script and choose a YOLO model from the available options.
4. **Background Removal**: Start the script to initiate real-time background removal on the live video feed.
5. **Quit**: Press 'Q' to exit the application.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics
- Pre-trained YOLO segmentation models
