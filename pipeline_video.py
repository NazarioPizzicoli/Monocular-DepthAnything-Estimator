import cv2
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from geopy.distance import great_circle
import math
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# 1. CONFIGURATION AND CONSTANTS
# ==============================================================================

# SYSTEM PATHS
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = DATA_DIR / "detection_video"

# INPUT SOURCES
SOURCE = "video_test.mp4" 
CAM_CSV_PATH = DATA_DIR / "preprocessed" / "camera_position_video.csv"
ANCHOR_CSV_PATH = DATA_DIR / "preprocessed" / "anchor_points_video.csv"
CALIB_YAML_PATH = DATA_DIR / "camera_calibration" / "calibration.yaml"

# MODEL PATHS
YOLO_WEIGHTS = MODELS_DIR / "YOLOv11" / "weights" / "yolo11n.pt"
DEPTH_ANYTHING_PATH = MODELS_DIR / "Depth-Anything-V2"

# DYNAMIC IMPORT OF CUSTOM MODELS
if str(DEPTH_ANYTHING_PATH) not in sys.path:
    sys.path.append(str(DEPTH_ANYTHING_PATH))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError as e:
    print(f"[ERROR] Unable to import DepthAnythingV2: {e}")
    sys.exit(1)

# DEVICE CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = "vitl" if torch.cuda.is_available() else "vits"

print(f"System initialized on: {DEVICE} correctly.")

# ==============================================================================
# 2. CLASSES
# ==============================================================================

class DepthEstimator:
    def __init__(self, model_path: Path, encoder: str, device: str):
        self.da_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        
        # Model loading
        print(f"Loading DA-V2 ({encoder})...")
        self.model = DepthAnythingV2(**self.da_configs[encoder])
        self.checkpoint = model_path / "checkpoints" / f"depth_anything_v2_{encoder}.pth"
        
        self.model.load_state_dict(torch.load(self.checkpoint, map_location='cpu'))
        self.model = self.model.to(device).eval()
        
        # Calibration parameters
        self.a = None
        self.b = None
    
    def predict(self, frame:np.ndarray) -> np.ndarray:
        """ Performs inference on a frame obtaining a relative depth map """
        return self.model.infer_image(frame)
        
    def calibrate(self, anchors_df:pd.DataFrame, rel_depth_map:np.ndarray) -> bool:
        """ Calculates a, b using visible anchor points. Returns True if correctly calculated. """
        h_map, w_map = rel_depth_map.shape
        rel_list = []
        dist_list = []
        
        for _, row in anchors_df.iterrows():
            ax, ay = int(row['x_pixel']), int(row['y_pixel'])
            gt_dist = row['distance_gt']
            
            if 0 <= ax < w_map and 0 <= ay < h_map:
                patch = rel_depth_map[max(0, ay-1):min(h_map, ay+2), max(0, ax-1):min(w_map, ax+2)]
                depth_rel = np.median(patch)
                rel_list.append(depth_rel)
                dist_list.append(gt_dist)
                
        if len(rel_list) < 2:
            return False
            
        # Least Squares: y = 1/D_metric, X = [D_rel, 1] inverse affine linear
        y = 1.0 / (np.array(dist_list) + 1e-12)
        X = np.vstack([np.array(rel_list), np.ones(len(rel_list))]).T
        
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.a, self.b = coeffs
        return True
    
    def get_metric_depth_at(self, rel_depth_map:np.ndarray, x:int, y:int) -> Tuple[float, float]:
        """ Converts relative depth to meters at point x,y (bounding box centroid) """
        if self.a is None or self.b is None:
            return -1.0, -1.0
        
        h, w = rel_depth_map.shape
        if not (0 <= x < w and 0 <= y <h):
            return -1.0, -1.0
        
        rel_depth_point = rel_depth_map[y, x]
        
        abs_depth_inv_point = self.a * rel_depth_point + self.b
        if abs_depth_inv_point <= 1e-6:
                return 999.9
        
        return 1.0 /abs_depth_inv_point, float(rel_depth_point)

class ObjectDetector:
    def __init__(self, weights_path: Path, device: str):
        self.device = device
        print(f"Loading YOLO ({weights_path.name})...")
        self.model = YOLO(weights_path)
        self.model.to(device)
        
    def detect(self, frame:np.ndarray) -> list[Dict]:
        """ Performs detection on a frame creating a list of dictionaries """
        results = self.model(frame, verbose=False, conf=0.5)
        if not results: return []
        
        clean_list = []
        result = results[0]
        for box in result.boxes:
            # 1. Bounding box coordinates
            coords = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, coords)
            
            # 2. Class (Label)
            cls_id = int(box.cls.cpu().numpy()[0])
            label_name = result.names[cls_id]

            # 3. Confidence
            confidence = float(box.conf.cpu().numpy()[0])

            detection_info = {
                'box': [x1, y1, x2, y2],
                'label': label_name,
                'conf': round(confidence, 2),
            }
            
            clean_list.append(detection_info)

        return clean_list

# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================

def haversine_3d(lat1:float, lon1:float, alt1:float, lat2:float, lon2:float, alt2:float) -> float:
    """ Calculates 3D metric distance between two points with the Haversine formula """
    dist_2D = great_circle((lat1, lon1), (lat2, lon2)).meters
    alt_diff = alt1 - alt2
    return math.sqrt(dist_2D**2 + alt_diff**2)

def save_detection_data(frame_img: np.ndarray, detections_data: List[Dict]):
    """
    Handles saving to the file system divided by days.
    detections_data is a list of dicts with: label, box, d_metric, d_rel
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    timestamp_filename = now.strftime("%H_%M_%S_%f")[:-3] # ms precision
    
    # 1. Daily Folders Creation
    daily_folder = OUTPUT_DIR / date_str
    frames_folder = daily_folder / "frames"
    csv_path = daily_folder / "detection_log.csv"
    
    frames_folder.mkdir(parents=True, exist_ok=True)
    
    # 2. Frame Saving (One per detection batch)
    # Filename: HH_MM_SS_ms.jpg
    frame_filename = f"{timestamp_filename}.jpg"
    cv2.imwrite(str(frames_folder / frame_filename), frame_img)
    
    # 3. CSV Update
    file_exists = csv_path.exists()
    
    with open(csv_path, "a") as f:
        if not file_exists:
            f.write("Date,Time,Frame_File,Label,Conf,BBox_x1,BBox_y1,BBox_x2,BBox_y2,Rel_Depth,Metric_Depth_m\n")
        
        for item in detections_data:
            x1, y1, x2, y2 = item['box']
            # Build CSV row
            row = (f"{date_str},{time_str},{frame_filename},"
                   f"{item['label']},{item['conf']},"
                   f"{x1},{y1},{x2},{y2},"
                   f"{item['d_rel']:.4f},{item['d_metric']:.2f}\n")
            f.write(row)

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================
            
def run_pipeline(csv_cam:Path, csv_anchor:Path, depth_path:Path, yolo_path:Path, encoder:str, device:str, source:str):
    # 1. Calculation and loading of geographic data
    print("Starting pipeline")
    if not csv_cam.exists() or not csv_anchor.exists():
        print("[ERROR]: CSV file not found.")
        return
    
    df_camera = pd.read_csv(csv_cam).iloc[0]
    df_anchors = pd.read_csv(csv_anchor)
    
    print("Calculating gt anchor distances...")
    distances = []
    for _ , anchor in df_anchors.iterrows():
        d = haversine_3d(df_camera['latitude'], df_camera['longitude'], df_camera['altitude'], anchor['latitude'], anchor['longitude'], anchor['altitude'])
        distances.append(d)
    
    df_anchors["distance_gt"] = distances
    
    # 2. Model initialization
    depth_esitmator = DepthEstimator(depth_path, encoder, device)
    object_detector = ObjectDetector(yolo_path, device)
    
    # 3. Video stream
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print("[ERROR] Video not opened")
        return
    
    print(f"Logs will be saved in: {OUTPUT_DIR}/")
    print("Streaming started. Press 'q' to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream lost")
                break
            
            display_frame = frame.copy()
            
            # A. Detection
            detections = object_detector.detect(frame)
            frame_log_data = []
            
            # B. Depth estimation
            if len(detections) > 0:
                rel_depth_map = depth_esitmator.predict(frame)
                
                # Calibration
                is_calibrated = depth_esitmator.calibrate(df_anchors, rel_depth_map)
                if is_calibrated:
                    for item in detections:
                        x1, y1, x2, y2 = item["box"]
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                       
                        d_metric, d_rel = depth_esitmator.get_metric_depth_at(rel_depth_map, cx, cy)
                        
                        label = f"{item['label']} {d_metric:.1f}m"
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.circle(display_frame, (cx, cy), 5, (0, 255, 255), -1)
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        item['d_metric'] = d_metric
                        item['d_rel'] = d_rel
                        frame_log_data.append(item)
                        
                    save_detection_data(display_frame, frame_log_data)
                else:
                    print(f" Calibration failed")
                    for item in detections:
                            x1, y1, x2, y2 = item["box"]
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # C. Visualization
            vis = cv2.resize(display_frame, (1280, 720))
                
            scale_x = 1280 / frame.shape[1]
            scale_y = 720 / frame.shape[0]
            for _, anchor in df_anchors.iterrows():
                ax, ay = int(anchor['x_pixel'] * scale_x), int(anchor['y_pixel'] * scale_y)
                cv2.circle(vis, (ax, ay), 3, (255, 0, 0), -1)

            cv2.imshow("RealTime Metric Depth", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("\nManual interruption.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == "__main__":
    run_pipeline(CAM_CSV_PATH,ANCHOR_CSV_PATH,DEPTH_ANYTHING_PATH,YOLO_WEIGHTS,ENCODER,DEVICE,SOURCE)