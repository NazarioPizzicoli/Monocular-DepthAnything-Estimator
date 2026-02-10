# RealTime Monocular Depth Estimation with Metric Calibration

A modular pipeline that transforms **Relative Depth Estimation (RDE)** from *Depth Anything V2* into accurate **Metric Depth Estimation (MDE)** in real-time. It integrates **YOLOv11** for object detection, providing distance measurements for every detected object using Ground Control Points (GCPs) calibration.

## Key Features

* **Zero-Shot Metricization:** Levers *Depth Anything V2* for relative depth and scales it to meters using GCPs.
* **Real-Time Object Detection:** Integrates *YOLOv11* to detect objects (e.g., boats, buoys) and label them with their real-world distance.
* **Live RTSP Streaming:** Full pipeline support for IP Cameras/RTSP streams with low-latency handling.
* **Calibration Tools:** dedicated scripts to extract pixel coordinates for anchors and record calibrated video datasets.

--

## Project Structure

```text
├── data/                  # Data directory (calibration, raw, results)
├── models/                # Place Depth-Anything-V2 and YOLO weights here
├── scripts/
│   ├── analysis/          # Plotting and video generation
│   ├── data_preparation/  # Synchronization of GPS/Video
│   ├── streaming/         # RTSP recording and coordinate extraction tools
├── realtime_pipeline.py   # MAIN SCRIPT: Live Inference + YOLO + MDE
├── requirements.txt       # Dependencies
└── README.md

--

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* NVIDIA GPU with CUDA support (Required for real-time performance)

### Setting up the Environment
```bash
git clone [https://github.com/NazarioPizzicoli/Monocular-DepthAnything-Estimator.git](https://github.com/NazarioPizzicoli/Monocular-DepthAnything-Estimator.git)
cd Monocular-DepthAnything-Estimator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Model Weights
* Place Depth Anything V2 weights in models/Depth-Anything-V2/checkpoints/
* Place YOLOv11 weights in models/YOLOv11/weights/


## Usage
### 1. Offline Analysis (Video + GPS Logs)
Synchronize video and GPS data, then run metric estimation analysis.
```bash
# Step 1: Preprocess (Sync Video & GPS)
python scripts/data_preparation/preprocess_and_match.py --exp 1
# Step 2: Generate Analysis Plots & Video
python scripts/analysis/generate_plots.py 1
```

### 2. Real-Time Streaming Pipeline (RTSP)
Run the full pipeline on a live video stream. This performs:
1. RTSP Capture (Low Latency)
2. Optical Undistortion
3. YOLOv11 Detection
4. Depth Anything V2 Inference
5. Metric Calibration (using Anchors)

```bash
python realtime_pipeline.py
```
Configuration: Check realtime_pipeline.py constants to set your RTSP URL/Video Path and model paths.

### 3. Utilities
Get Anchor Coordinates:
Click on the video stream to get (x, y) pixel coordinates for your anchor_points.csv.
```bash
python scripts/streaming/stream_coords.py```


#TODO

1. Modify video source or RTSP
2. Modify camera_position_stream.csv, anchor_points_stream.csv, (or camera_position_video.csv, anchor_points_video.csv in case a video is used), calibration.yaml (video) HAS TO BE 1FPS)
3. Modify yolo weight
4. Download yolo e depthanything
