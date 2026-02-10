# RealTime Monocular Depth Estimation with Metric Calibration

This modular pipeline transforms **Relative Depth Estimation (RDE)** from *Depth Anything V2* into accurate **Metric Depth Estimation (MDE)** in real-time. It integrates **YOLOv11** for object detection, providing distance measurements for every detected object using Ground Control Points (GCP) calibration.

<p align="center" width="100%">
    <img src="assets/pipeline.gif" width="800" alt="Object detection + Depth Estimation drone in a lake.">
</p>

## Installation and Requirements

### Prerequisites
* Python 3.8+
* NVIDIA GPU with CUDA support (strongly recommended for real-time performance).

### Environment Setup
```bash
git clone [https://github.com/NazarioPizzicoli/Monocular-DepthAnything-Estimator.git](https://github.com/NazarioPizzicoli/Monocular-DepthAnything-Estimator.git)
cd Monocular-DepthAnything-Estimator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Model Weights
You must download the model weights and place them in the following directories:
1. Depth Anything V2: Download checkpoints from the Official Repository and place them in models/Depth-Anything-V2/checkpoints/.
2. YOLOv11: Download weights (e.g., yolo11n.pt) from Ultralytics Documentation and place them in models/YOLOv11/weights/.

## Data Configuration
The system relies on specific configuration files located in data/preprocessed/ and data/camera_calibration/. The provided files are templates and must be modified by the user.

In case of stream:
- Modify "calibration.yaml" with the real camera intrinsic parameters for undistortion.
- Modify "camera_position_stream.csv" with the real GPS coordinates of the camera.
- Modify "anchor_points_stream.csv" with the real GPS and pixel coordinates for Ground Control Points (look step 3. Coordinate Extraction Utility).
- Modify "SOURCE" with the correct RTSP URL in pipeline_stream.py

In case of recorded video:
- Modify "calibration.yaml" with the real camera intrinsic parameters for undistortion.
- Modify "camera_position_video.csv" with the real GPS coordinates of the camera.
- Modify "anchor_points_video.csv" with the real GPS and pixel coordinates for Ground Control Points (look step 3. Coordinate Extraction Utility).
- Upload a recorded video as source and call it "video_test.mp4".

## Usage
### 1. Offline Video Analysis
Run the pipeline on a local video file. Note: This mode assumes the video is already oriented correctly and undistorted.
```bash
python pipeline_video.py
```

### 2. Real-Time Streaming (RTSP)
Run the pipeline on a live RTSP stream. This includes 180° rotation and optical undistortion.
```bash
python pipeline_stream.py
```

### 3. Coordinate Extraction Utility
To easily extract pixel coordinates for your anchor points, run this utility and click on the desired points in the video stream:
```bash
python scripts/utils/stream_coords.py
```
