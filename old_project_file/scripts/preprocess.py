#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# esempio: python3 /media/naza/SBT2025/pipeline/scripts/preprocess.py -e 1 -p naza -r SBT2025

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Video–GPS preprocessing pipeline")

    parser.add_argument(
        "-e", "--exp",
        type=str,
        default="0",
        help="ID dell'esperimento: 1, 2, 3, 4 oppure 0 per saltare il preprocessing"
    )
    """
    parser.add_argument(
        "-p", "--pc",
        type=str,
        default="naza",
        help="Nome del pc: naza o labust"
    )

    parser.add_argument(
        "-r", "--root",
        type=str,
        default="SBT2025",
        help="Nome del disco: SBT2025 o test_ssd"
    )
    """
    return parser.parse_args()

# ========================================================================
# MAIN ENTRY POINT
# ========================================================================
def main():
    args = parse_args()

    EXP_ID = args.exp
    """
    PC = args.pc
    root = args.root

    print(f"\nCONFIGURAZIONE:")
    print(f"  EXP_ID: {EXP_ID}")
    print(f"  PC:     {PC}")
    print(f"  root:   {root}")
    print("=" * 60)
    """
    # Try importing MCAP
    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call(['pip', 'install', 'mcap', 'mcap-ros2-support'])
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory

    # Experiment dictionary
    PREPROCESS_DICT = {
        "1": dict(video_raw="video_c_20251106_152844.mp4",
                  mcap="gps_20251106_153635.mcap",
                  label="label_152844",
                  name="exp1_152844"),
        "2": dict(video_raw="video_c_20251106_154149.mp4",
                  mcap="gps_20251106_154245.mcap",
                  label="label_154149",
                  name="exp2_154149"),
        "3": dict(video_raw="video_c_20251106_161904.mp4",
                  mcap="gps_20251106_161835.mcap",
                  label="label_161904",
                  name="exp3_161904"),
        "4": dict(video_raw="video_c_20251106_162533.mp4",
                  mcap="gps_20251106_162531.mcap",
                  label="label_162533",
                  name="exp4_162533"),
    }

    if EXP_ID == "0":
        print("\n  Preprocessing SKIPPED (EXP_ID = '0')")
        print("=" * 60)
        return

    if EXP_ID not in PREPROCESS_DICT:
        raise ValueError(f"Invalid EXP_ID: {EXP_ID}. Choose from {list(PREPROCESS_DICT.keys())}")

    EXP = PREPROCESS_DICT[EXP_ID]

    # Paths
    DATA_ROOT = Path(__file__).parent.parent
    VIDEO_PATH = DATA_ROOT / "video_raw" / EXP["video_raw"]
    MCAP_PATH = DATA_ROOT / "mcap" / EXP["mcap"]
    LABELS_DIR = DATA_ROOT / "label" / EXP["label"]

    # Outputs
    FRAMES_OUTPUT = DATA_ROOT / "preprocessed" / f"frames_{EXP['name']}"
    GPS_CSV_OUTPUT = DATA_ROOT / "preprocessed" / f"gps_{EXP['name']}.csv"
    VIDEO_1FPS_OUTPUT = DATA_ROOT / "preprocessed" / f"video_1fps_{EXP['name']}.mp4"

    FRAMES_OUTPUT.mkdir(exist_ok=True, parents=True)

    # Parameters
    TARGET_FPS = 1
    GPS_TOPIC = "/cres/ublox_gps_node/fix"
    TIME_TOLERANCE_MS = 100

    print(f"\nExperiment: {EXP['name']}")
    print(f"Input video: {VIDEO_PATH.name}")
    print(f"Input MCAP: {MCAP_PATH.name}")
    print(f"Target FPS: {TARGET_FPS}")
    
    # =========================================================================
    # EXTRA: Create camera_position.csv and anchor_points.csv
    # =========================================================================
    print("\n[EXTRA] Creating camera_position.csv and anchor_points.csv...")

    camera_position_path = DATA_ROOT / "preprocessed" / "camera_position.csv"
    anchor_points_path = DATA_ROOT / "preprocessed" / "anchor_points.csv"

    # Camera position (single static record)
    df_cam = pd.DataFrame([{
        "latitude": 45.7839523,
        "longitude": 15.9098636,
        "altitude": 161.334
    }])
    df_cam.to_csv(camera_position_path, index=False)

    # Anchor points table
    df_anchor = pd.DataFrame([
        {
            "anchor_id": "pontile_vicino",
            "latitude": 45.784222,
            "longitude": 15.9102956,
            "altitude": 159.411,
            "x_pixel": 826,
            "y_pixel": 520
        },
        {
            "anchor_id": "pontile_lontano",
            "latitude": 45.7844181,
            "longitude": 15.9104346,
            "altitude": 159.409,
            "x_pixel": 609,
            "y_pixel": 501
        },
        {
            "anchor_id": "boa",
            "latitude": 45.7841453,
            "longitude": 15.9106661,
            "altitude": 159.418,
            "x_pixel": 1445,
            "y_pixel": 479
        }
    ])
    df_anchor.to_csv(anchor_points_path, index=False)

    print(f"  ✓ camera_position.csv saved to: {camera_position_path}")
    print(f"  ✓ anchor_points.csv saved to: {anchor_points_path}")

    # =========================================================================
    # 1. Parse video metadata
    # =========================================================================
    print("\n[1/6] Parsing video metadata...")

    video_name = VIDEO_PATH.stem
    parts = video_name.split('_')
    date_str = parts[2]
    time_str = parts[3]

    video_start_dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    video_start_timestamp = video_start_dt.timestamp()

    print(f"  Video start: {video_start_dt}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video: {video_fps} fps, {total_frames} frames, {img_width}x{img_height}")

    # =========================================================================
    # 2. Extract GPS data from MCAP
    # =========================================================================
    print("\n[2/6] Extracting GPS data from MCAP...")

    gps_data = []

    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=[GPS_TOPIC]
        ):
            stamp = ros_msg.header.stamp
            msg_timestamp = stamp.sec + stamp.nanosec * 1e-9

            gps_data.append({
                'timestamp': msg_timestamp,
                'latitude': ros_msg.latitude,
                'longitude': ros_msg.longitude,
                'altitude': ros_msg.altitude,
                'status': ros_msg.status.status,
                'service': ros_msg.status.service
            })

    df_gps = pd.DataFrame(gps_data)
    df_gps = df_gps.sort_values('timestamp').reset_index(drop=True)

    print(f"  Extracted {len(df_gps)} GPS messages")

    df_gps_valid = df_gps[df_gps['status'] >= 0].copy()

    # =========================================================================
    # 3. Extract frames at 1 FPS
    # =========================================================================
    print("\n[3/6] Extracting frames at 1 fps...")

    frame_interval = int(video_fps / TARGET_FPS)
    frame_data = []
    frame_count = 0
    extracted_count = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pbar = tqdm(total=total_frames, desc="  Processing video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_time_offset = frame_count / video_fps
            frame_timestamp = video_start_timestamp + frame_time_offset

            time_diffs = np.abs(df_gps_valid['timestamp'] - frame_timestamp)
            closest_idx = time_diffs.idxmin()
            closest_gps = df_gps_valid.loc[closest_idx]
            time_diff_ms = time_diffs.min() * 1000

            if time_diff_ms <= TIME_TOLERANCE_MS:
                frame_name = f"frame_{datetime.fromtimestamp(frame_timestamp).strftime('%H%M%S')}.png"
                frame_path = FRAMES_OUTPUT / frame_name
                cv2.imwrite(str(frame_path), frame)

                frame_data.append({
                    'frame_name': frame_name,
                    'timestamp': datetime.fromtimestamp(frame_timestamp).strftime('%H:%M:%S'),
                    'timestamp_unix': frame_timestamp,
                    'latitude': closest_gps['latitude'],
                    'longitude': closest_gps['longitude'],
                    'altitude': closest_gps['altitude'],
                    'gps_time_diff_ms': time_diff_ms,
                    'x_pixel': np.nan,
                    'y_pixel': np.nan
                })
                extracted_count += 1

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"  Extracted {extracted_count} frames")

    # =========================================================================
    # 4. Load YOLO labels
    # =========================================================================
    print("\n[4/6] Loading YOLO labels...")

    if LABELS_DIR.exists():
        for frame_info in frame_data:
            label_path = LABELS_DIR / frame_info['frame_name'].replace('.png', '.txt')

            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                if lines:
                    parts = lines[0].split()
                    if len(parts) >= 5:
                        x_norm, y_norm = float(parts[1]), float(parts[2])
                        frame_info['x_pixel'] = x_norm * img_width
                        frame_info['y_pixel'] = y_norm * img_height
    else:
        print("  Labels directory not found, skipping YOLO annotations.")

    # =========================================================================
    # 5. Save GPS CSV
    # =========================================================================
    print("\n[5/6] Saving GPS CSV...")

    df_target = pd.DataFrame(frame_data)
    df_target.to_csv(GPS_CSV_OUTPUT, index=False)

    print(f"  GPS CSV saved to: {GPS_CSV_OUTPUT}")

    # =========================================================================
    # 6. Create 1 FPS video
    # =========================================================================
    print("\n[6/6] Creating 1 fps video from extracted frames...")

    frame_files = sorted(FRAMES_OUTPUT.glob("*.png"))

    if frame_files:
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]

        out = cv2.VideoWriter(
            str(VIDEO_1FPS_OUTPUT),
            cv2.VideoWriter_fourcc(*'mp4v'),
            1,
            (width, height)
        )

        for frame_path in tqdm(frame_files, desc="  Writing video"):
            frame = cv2.imread(str(frame_path))
            out.write(frame)

        out.release()
        print(f"  Video saved: {VIDEO_1FPS_OUTPUT}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Preprocessing complete ✓")
    print(f"  Frames: {extracted_count}")
    print(f"  GPS records: {len(df_target)}")
    print(f"  Avg sync error: {df_target['gps_time_diff_ms'].mean():.2f} ms")
    print(f"  Max sync error: {df_target['gps_time_diff_ms'].max():.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
