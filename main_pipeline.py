import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
import gc
from pathlib import Path
from tqdm import tqdm
from geopy.distance import great_circle 
from sklearn.metrics import mean_squared_error, mean_absolute_error # Importa metriche necessarie

# ==============================================================================
# --- 0. Setup & Configurazione ---
# ==============================================================================

DATA_ROOT = Path(__file__).parent 
DEPTH_ANYTHING_DIR = DATA_ROOT / "models" / "Depth-Anything-V2"

if str(DEPTH_ANYTHING_DIR) not in sys.path:
    sys.path.append(str(DEPTH_ANYTHING_DIR))

try:
    # Assicurati che l'importazione funzioni correttamente se la struttura è come previsto
    from depth_anything_v2.dpt import DepthAnythingV2
    print(" DepthAnythingV2 importato con successo.") 
except ImportError:
    print("ERRORE: Impossibile importare DepthAnythingV2. Assicurati che 'models/Depth-Anything-V2' esista e contenga il codice necessario.")
    sys.exit(1)

# Configurazioni del modello
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENCODER = 'vits' if DEVICE == 'cpu' else 'vitl' 
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}


# ==============================================================================
# --- 1. Funzioni di Utilità (Integrated for simplicity) ---
# ==============================================================================

def haversine_3d(lat1, lon1, alt1, lat2, lon2, alt2):
    """Calcola la distanza 3D (GT) tra due punti geografici in metri."""
    horizontal_dist = great_circle((lat1, lon1), (lat2, lon2)).meters
    vertical_dist = alt2 - alt1 # Assumendo altitudini in metri
    return np.sqrt(horizontal_dist**2 + vertical_dist**2)

def normalize_depth_map(depth_map):
    """Normalizza la mappa di profondità a un range [0, 1]."""
    min_val = depth_map.min()
    max_val = depth_map.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(depth_map)
    normalized = (depth_map - min_val) / (max_val - min_val)
    return normalized

def get_depth_map_scale(original_img_path, depth_map):
    """Calcola il fattore di scala tra immagine originale e depth map."""
    img = cv2.imread(str(original_img_path))
    if img is None:
        return 1.0, 1.0
    orig_h, orig_w = img.shape[:2]
    depth_h, depth_w = depth_map.shape
    return depth_w / orig_w, depth_h / orig_h

def extract_depth_at_point(depth_map, x, y, scale_x=1.0, scale_y=1.0, patch_size=3):
    """Estrae la profondità (mediana) da una patch attorno al punto (con scaling)."""
    x_scaled = int(x * scale_x)
    y_scaled = int(y * scale_y)
    
    h, w = depth_map.shape
    half = patch_size // 2
    
    y_start = max(0, y_scaled - half)
    y_end = min(h, y_scaled + half + 1)
    x_start = max(0, x_scaled - half)
    x_end = min(w, x_scaled + half + 1)
    
    patch = depth_map[y_start:y_end, x_start:x_end]
    return np.median(patch) if patch.size > 0 else np.nan

def calibrate_depth(df_anchors, distances_to_anchors, depths_at_anchors):
    """
    Esegue la calibrazione metrica utilizzando i GCPs (anchor points) con il modello inverso.
    Modello: 1/D_metric = a * D_relative + b
    """
    anchor_relative = []
    anchor_metric = []
    
    # Riunisce tutti i campioni validi (tutti gli anchor su tutti i frame)
    for frame_idx in range(depths_at_anchors.shape[0]):
        for anchor_idx in range(depths_at_anchors.shape[1]):
            rel_depth = depths_at_anchors[frame_idx, anchor_idx]
            met_dist = distances_to_anchors[anchor_idx]
            
            if not np.isnan(rel_depth) and rel_depth > 0:
                anchor_relative.append(rel_depth)
                anchor_metric.append(met_dist)

    anchor_relative = np.array(anchor_relative)
    anchor_metric = np.array(anchor_metric)
    
    if len(anchor_relative) < 2:
        print("ERRORE: Campioni insufficienti per la calibrazione.")
        return None, None

    # Implementazione Least Squares per il modello inverso (y = a*x + b)
    # y = 1 / D_metric, x = D_relative
    y = 1.0 / (anchor_metric + 1e-12)
    X = np.vstack([anchor_relative, np.ones_like(anchor_relative)]).T
    
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a_opt, b_opt = coeffs
    
    # Funzione di conversione calibrata: D_metric = 1 / (a * D_relative + b)
    def relative_to_metric(depth_relative, a=a_opt, b=b_opt):
        eps = 1e-12
        return 1.0 / (a * depth_relative + b + eps)
    
    distances_pred = relative_to_metric(anchor_relative)
    rmse = np.sqrt(mean_squared_error(anchor_metric, distances_pred))
    mae = mean_absolute_error(anchor_metric, distances_pred)
    mape = np.mean(np.abs((distances_pred - anchor_metric) / anchor_metric)) * 100
    
    print("\n--- Calibration Results ---")
    print(f"  Calib. Samples: {len(anchor_relative)}")
    print(f"  Coeff. a: {a_opt:.6f}, b: {b_opt:.6f}")
    print(f"  RMSE (Calib. Data): {rmse:.2f} m")
    print(f"  MAE (Calib. Data): {mae:.2f} m")
    print(f"  MAPE (Calib. Data): {mape:.2f} %")
    print("----------------------------")

    return relative_to_metric, {'a': a_opt, 'b': b_opt, 'rmse': rmse, 'mae': mae}

# ==============================================================================
# --- 2. Pipeline Principale ---
# ==============================================================================

def run_depth_pipeline(exp_id: str, reload_depth_maps: bool = False):
    
    print(f"--- Running Real-Time MDE Pipeline for EXP {exp_id} ---")
    print("=" * 60)

    # --- 2.1 Configurazione Percorsi ---
    EXPERIMENT_DICT = {
        "1": dict(name="exp1_152844"),
        "2": dict(name="exp2_154149"),
        "3": dict(name="exp3_161904"),
        "4": dict(name="exp4_162533"),
    }
    
    if exp_id not in EXPERIMENT_DICT:
        raise ValueError(f"Invalid experiment ID: {exp_id}")
    
    EXP = EXPERIMENT_DICT[exp_id]
    
    # Pathnames based on MDE-GitHub/data structure
    DATA_DIR = Path(__file__).parent / "data"
    FRAMES_PATH = DATA_DIR / "preprocessed" / f"frames_{EXP['name']}"
    TARGET_CSV = DATA_DIR / "preprocessed" / f"gps_{EXP['name']}.csv"
    CAMERA_CSV = DATA_DIR / "preprocessed" / "camera_position.csv"
    ANCHORS_CSV = DATA_DIR / "preprocessed" / "anchor_points.csv"
    
    DEPTH_OUTPUT_DIR = DATA_DIR / "depth_map_dav2" / f"relative_{EXP['name']}"
    RESULTS_DIR = DATA_DIR / "results_dav2" / f"results_{EXP['name']}"
    
    DEPTH_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"Experiment: {EXP['name']}, Device: {DEVICE}, Encoder: {ENCODER}")
    
    # --- 2.2 Caricamento Dati & Calcolo Distanze GT ---
    if not TARGET_CSV.exists() or not ANCHORS_CSV.exists() or not CAMERA_CSV.exists():
        print(f"ERRORE: File CSV pre-elaborati non trovati in {TARGET_CSV.parent}.")
        print("Eseguire prima 'scripts/data_preparation/preprocess_and_match.py'.")
        return

    df_camera = pd.read_csv(CAMERA_CSV)
    df_target = pd.read_csv(TARGET_CSV)
    df_anchors = pd.read_csv(ANCHORS_CSV)
    cam = df_camera.iloc[0]
    
    print("\n[STEP 1/3] Calcolo distanze Ground Truth (GT)...")
    
    distances_to_anchors = np.array([
        haversine_3d(
            cam['latitude'], cam['longitude'], cam['altitude'],
            anchor['latitude'], anchor['longitude'], anchor['altitude']
        )
        for _, anchor in df_anchors.iterrows()
    ])
    
    distances_to_target = df_target.apply(
        lambda row: haversine_3d(
            cam['latitude'], cam['longitude'], cam['altitude'],
            row['latitude'], row['longitude'], row['altitude']
        ) if pd.notna(row['x_pixel']) else np.nan,
        axis=1
    ).values
    
    print(f"  Distanze Anchor (GT): {distances_to_anchors.round(2)} m")
    print(f"  Frame Target validi: {df_target['x_pixel'].notna().sum()}/{len(df_target)}")

    # --- 2.3 Inferenza MDE ---
    print("\n[STEP 2/3] Stima Profondità Relativa (Depth Anything V2)...")
    
    ModelClass = DepthAnythingV2 # Usiamo la classe importata
    
    model = None
    if not reload_depth_maps:
        print(f"  Caricamento Modello {ENCODER}...")
        model = ModelClass(**MODEL_CONFIGS[ENCODER])
        checkpoint_path = DEPTH_ANYTHING_DIR / 'checkpoints' / f"depth_anything_v2_{ENCODER}.pth"
        if not checkpoint_path.exists():
            print(f"ERRORE: Pesi del modello non trovati in {checkpoint_path}")
            print("  Scaricare i pesi e posizionarli in checkpoints/")
            return
        
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model = model.to(DEVICE).eval()
        print("  Modello caricato.")

    frame_names = df_target['frame_name'].tolist()
    depths_at_target = []
    depths_at_anchors = np.zeros((len(frame_names), len(df_anchors))) * np.nan
    
    for frame_idx, frame_name in enumerate(tqdm(frame_names, desc="  Processing Frames")):
        frame_path = FRAMES_PATH / frame_name
        # Nomi dei file di cache
        normalized_cache_path = DEPTH_OUTPUT_DIR / frame_name.replace('.png', '_depth_normalized.npy')
        cache_path = DEPTH_OUTPUT_DIR / frame_name.replace('.png', '_depth.npy')
        
        depth_normalized = None
        
        # 1. Tenta di ricaricare la mappa normalizzata
        if normalized_cache_path.exists():
            depth_normalized = np.load(normalized_cache_path)
        
        # 2. Se non ricaricata, calcola/ricarica/normalizza
        if depth_normalized is None:
            depth = None

            if cache_path.exists():
                depth = np.load(cache_path)
            elif model is not None:
                img = cv2.imread(str(frame_path))
                if img is None:
                    print(f"Warning: Cannot load {frame_name}")
                    depths_at_target.append(np.nan)
                    depths_at_anchors[frame_idx, :] = np.nan
                    continue
                
                # Inferenza
                with torch.no_grad():
                    # model.infer_image() deve essere una funzione definita nel modulo DPT/DepthAnythingV2
                    # Qui assumiamo che prenda l'immagine e restituisca la mappa di profondità raw
                    # Il notebook originale usava un preprocessore e un infer() separati.
                    # Per l'uso in uno script, dobbiamo assicurarci che il modello gestisca il pre-processing interno.
                    
                    # Se non è definita una funzione infer_image(), l'utente dovrà usare:
                    # from depth_anything_v2.util.transform import TransformCV2
                    # image_tensor = TransformCV2.apply_transform(img)
                    # depth = model(image_tensor.to(DEVICE)).cpu().numpy().squeeze() 
                    
                    # Manteniamo una chiamata generica, l'utente dovrà adattare il metodo inferenza
                    try:
                        depth = model.infer_image(img) 
                    except AttributeError:
                        print("Warning: model.infer_image non trovato. Usare Transform e forward pass.")
                        # Placeholder: assumi che l'utente adatti la logica di inferenza qui.
                        depths_at_target.append(np.nan)
                        depths_at_anchors[frame_idx, :] = np.nan
                        continue
                        
                np.save(cache_path, depth)
                del img
            else:
                if reload_depth_maps:
                    pass 
                else:
                    raise RuntimeError(f"Model not loaded but cache missing for {frame_name}")

            if depth is not None:
                depth_normalized = normalize_depth_map(depth)
                np.save(normalized_cache_path, depth_normalized)
                del depth
        
        if depth_normalized is None:
            depths_at_target.append(np.nan)
            depths_at_anchors[frame_idx, :] = np.nan
            continue
            
        # Estrazione profondità target e anchor
        scale_x, scale_y = get_depth_map_scale(frame_path, depth_normalized)
        
        target_row = df_target[df_target['frame_name'] == frame_name].iloc[0]
        
        # Target
        if pd.notna(target_row['x_pixel']):
            depth_val = extract_depth_at_point(
                depth_normalized, 
                target_row['x_pixel'], 
                target_row['y_pixel'],
                scale_x, scale_y
            )
            depths_at_target.append(depth_val)
        else:
            depths_at_target.append(np.nan)
        
        # Anchors
        for anchor_idx, (_, anchor) in enumerate(df_anchors.iterrows()):
            if pd.notna(anchor['x_pixel']) and pd.notna(anchor['y_pixel']):
                depth_val = extract_depth_at_point(
                    depth_normalized, 
                    anchor['x_pixel'], 
                    anchor['y_pixel'],
                    scale_x, scale_y
                )
                depths_at_anchors[frame_idx, anchor_idx] = depth_val

        del depth_normalized
        if (frame_idx + 1) % 50 == 0:
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

    depths_at_target = np.array(depths_at_target)
    
    if model is not None:
        del model
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        
    print(f"\n  Profondità target estratte: {np.sum(~np.isnan(depths_at_target))}/{len(depths_at_target)} frames validi.")


    # --- 2.4 Calibrazione e Proiezione Metrica ---
    print("\n[STEP 3/3] Calibrazione Metrica e Stima Profondità Target...")
    
    # Rimuovi i frame che non hanno profondità valida per *tutti* gli anchor
    valid_anchor_frames = np.sum(~np.isnan(depths_at_anchors), axis=1) == len(df_anchors)
    
    if valid_anchor_frames.sum() == 0:
        print("AVVISO: Nessun frame contiene dati di profondità validi per tutti gli anchor. Impossibile calibrare.")
        return

    calibrated_depths_at_anchors = depths_at_anchors[valid_anchor_frames, :]
    
    # La calibrazione viene eseguita solo sui dati validi
    relative_to_metric_func, calib_coeffs = calibrate_depth(
        df_anchors, distances_to_anchors, calibrated_depths_at_anchors
    )
    
    if relative_to_metric_func is None:
        return

    # Applica la calibrazione al target
    target_distances_pred = np.array([
        relative_to_metric_func(d) if not np.isnan(d) else np.nan 
        for d in depths_at_target
    ])

    # Calcola le metriche finali sul target
    valid_mask = ~np.isnan(target_distances_pred) & ~np.isnan(distances_to_target)
    if valid_mask.sum() > 0:
        target_errors = target_distances_pred[valid_mask] - distances_to_target[valid_mask]
        target_rmse = np.sqrt(np.mean(target_errors**2))
        target_mae = np.mean(np.abs(target_errors))
        
        print("\n--- Target Prediction Metrics ---")
        print(f"  RMSE: {target_rmse:.2f} m")
        print(f"  MAE:  {target_mae:.2f} m")
        print(f"  Valid predictions: {valid_mask.sum()}/{len(distances_to_target)}")
        print("---------------------------------")

    # --- 2.5 Salvataggio Risultati ---
    results_df = df_target.copy()
    results_df['depth_relative'] = depths_at_target
    results_df['distance_predicted'] = target_distances_pred
    results_df['distance_groundtruth'] = distances_to_target
    results_df['error'] = target_distances_pred - distances_to_target
    results_df['abs_error'] = np.abs(results_df['error'])
    
    output_csv = RESULTS_DIR / f"results_{EXP['name']}.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nRISULTATI SALVATI: {output_csv}")
    
    print("\n" + "=" * 60)
    print("Pipeline di Inferenza e Calibrazione Completata.")
    print("Eseguire 'scripts/analysis/generate_plots.py' per l'analisi finale.")
    print("=" * 60)

    
# ==============================================================================
# --- MAIN EXECUTION POINT ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Real-Time Monocular Depth Estimation Pipeline (Inference & Metric Calibration).")
    parser.add_argument('exp_id', type=str, choices=['1', '2', '3', '4'], 
                        help="ID dell'esperimento da eseguire (corrispondente al file gps_expID_*.csv).")
    parser.add_argument('--reload', action='store_true', 
                        help="Se presente, ricarica le mappe di profondità da cache invece di eseguire l'inferenza del modello.")
    
    args = parser.parse_args()
    
    run_depth_pipeline(args.exp_id, args.reload)


if __name__ == "__main__":
    main()