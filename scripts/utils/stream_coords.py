#!/usr/bin/env python3
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import os
import argparse

"""
Nome: stream.py
Obiettivo: Streaming
Funzionalità:
- Caricamento matrice di calibrazione;
- Inizializzazione per low-latency;
- Salvataggio frame e stream (OPZIONALE);
- Recupero coordinate pixel al click (NUOVO);
"""

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
RTSP_URL = "rtsp://admin:labos023@10.0.250.179:554/Streaming/Channels/101"
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "scripts" else Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "video_raw"
CALIBRATION_FILE = PROJECT_ROOT / "data" / "calibration" / "calibration.yaml"

ENABLE_RECORDING = False
ENABLE_COORD = True  # <--- NUOVO FLAG PER ABILITARE IL CLICK

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FUNZIONI
# ============================================================================
def load_calibration(path: Path):
    """Carica la calibrazione da un file YAML OpenCV."""
    if not path.exists():
        raise FileNotFoundError(f"File di calibrazione non trovato: {path}")
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    if camera_matrix is None or dist_coeffs is None:
        raise RuntimeError("File di calibrazione non valido o nodi mancanti.")
    return camera_matrix, dist_coeffs


def init_camera(rtsp_url: str, low_latency=True):
    """Inizializza la cattura video da RTSP con opzioni per ridurre la latenza."""
    if low_latency:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;0"
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire lo stream RTSP")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream aperto: {width}x{height} @ {fps:.2f} FPS")
    return cap, fps, width, height


def setup_output(fps, width, height, record=True):
    """Crea la cartella di salvataggio e inizializza il VideoWriter."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = OUTPUT_DIR / f"video_c_{timestamp}"
    folder_path.mkdir(parents=True, exist_ok=True)
    
    out = None
    video_path = None
    
    if record:
        video_path = folder_path / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        print(f"[INFO] Registrazione video ATTIVA: {video_path}")
    else:
        print(f"[INFO] Registrazione video DISATTIVATA. Cartella sessione: {folder_path}")
        
    return folder_path, video_path, out, timestamp


def mouse_callback(event, x, y, flags, param):
    """
    Gestisce il click del mouse.
    param: dizionario contenente i fattori di scala {'scale_x', 'scale_y'}
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        scale_x = param['scale_x']
        scale_y = param['scale_y']
        
        # Calcola le coordinate reali sull'immagine originale (non quella ridimensionata per il display)
        real_x = int(x * scale_x)
        real_y = int(y * scale_y)
        
        print(f"[COORD] Click Display ({x}, {y}) -> Frame Reale ({real_x}, {real_y})")


def record_loop(cap, out, new_camera_mtx, roi, camera_matrix, dist_coeffs, folder_path, timestamp):
    """Loop principale di acquisizione e salvataggio."""
    x_roi, y_roi, w_roi, h_roi = roi
    record_start = time.time()
    last_ok = record_start

    print("[INFO] Premi 'c' per salvare frame, 'q' per uscire.")
    
    # Setup finestra e callback mouse se abilitato
    window_name = "Calibrated Stream"
    # Dimensione fissa del display (come definito nel resize successivo)
    display_w, display_h = 960, 540 
    
    if ENABLE_COORD:
        cv2.namedWindow(window_name)
        # Calcoliamo il fattore di scala tra il display e il frame reale (ritagliato dalla ROI)
        scale_params = {
            "scale_x": w_roi / display_w,
            "scale_y": h_roi / display_h
        }
        cv2.setMouseCallback(window_name, mouse_callback, scale_params)
        print("[INFO] Click mouse ABILITATO: clicca sullo stream per le coordinate.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if time.time() - last_ok > 2:
                print("[WARN] Frame perso, riconnessione...")
                cap.release()
                time.sleep(1)
                cap.open(RTSP_URL)
                continue
            continue

        last_ok = time.time()

        # Correzione ottica e rotazione
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame_undist = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
        frame_calibrated = frame_undist[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

        if out is not None:
            out.write(frame_calibrated)

        # Mostra anteprima
        display = cv2.resize(frame_calibrated, (display_w, display_h))
        cv2.imshow(window_name, display)

        # Tasti
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            elapsed = time.time() - record_start
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            ms = int((elapsed * 1000) % 1000)

            frame_filename = folder_path / f"frame_{h:02d}h_{m:02d}m_{s:02d}s_{ms:03d}ms.png"
            cv2.imwrite(str(frame_filename), frame_calibrated)
            print(f"[INFO] Frame salvato ({elapsed:.2f}s): {frame_filename.name}")

        elif key == ord('q'):
            print("[INFO] Chiusura richiesta.")
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Registrazione completata: {folder_path}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("        STREAM CALIBRATO - REGISTRAZIONE RTSP")
    print(f"        MODALITA' REGISTRAZIONE: {'ON' if ENABLE_RECORDING else 'OFF'}")
    print(f"        MODALITA' COORDINATE:    {'ON' if ENABLE_COORD else 'OFF'}")
    print("="*70)
    
    try:
        camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)
    except Exception as e:
        print(f"[ERROR] {e}")
        exit(1)
        
    cap, fps, width, height = init_camera(RTSP_URL)

    # Frame di test per calcolo ROI
    ret, test_frame = cap.read()
    if not ret:
        raise RuntimeError("Errore nel leggere il primo frame.")
    test_frame = cv2.rotate(test_frame, cv2.ROTATE_180)
    h, w = test_frame.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Passiamo il flag ENABLE_RECORDING a setup_output
    folder_path, video_path, out, timestamp = setup_output(fps, roi[2], roi[3], record=ENABLE_RECORDING)
    if ENABLE_RECORDING:
        print(f"[INFO] Salvataggio in: {video_path}")
        
    record_loop(cap, out, new_camera_mtx, roi, camera_matrix, dist_coeffs, folder_path, timestamp)