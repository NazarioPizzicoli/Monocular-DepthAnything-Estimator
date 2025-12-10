import argparse
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ==============================================================================
# --- Configurazione e Setup ---
# ==============================================================================
DATA_ROOT = Path(__file__).parent.parent.parent

# Mappatura della colormap per la profondità (simulazione di cv2.COLORMAP_JET)
def apply_colormap(depth_map_normalized):
    """Applica una colormap (simulata JET) a una mappa di profondità normalizzata [0, 1]."""
    depth_8bit = (depth_map_normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

def generate_plots(df_results, output_dir, exp_name):
    """Genera i grafici di analisi: serie temporale e istogramma degli errori."""
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Serie Temporale: Distanza Predetta vs. Ground Truth
    plt.figure(figsize=(15, 6))
    plt.plot(df_results['timestamp_unix'], df_results['distance_groundtruth'], 
             label='Distanza Ground Truth (GT)', color='green', linewidth=2)
    plt.plot(df_results['timestamp_unix'], df_results['distance_predicted'], 
             label='Distanza Predetta (MDE Calibrato)', color='red', linestyle='--')
    
    plt.title(f'Esp. {exp_name}: Stima di Profondità vs. Tempo')
    plt.xlabel('Timestamp UNIX')
    plt.ylabel('Distanza (metri)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "time_series_depth.png")
    plt.close()
    print(f"  ✓ Grafico Serie Temporale salvato.")
    
    # 2. Istogramma degli Errori Assoluti
    valid_errors = df_results['abs_error'].dropna()
    plt.figure(figsize=(8, 6))
    plt.hist(valid_errors, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(valid_errors.mean(), color='red', linestyle='dashed', linewidth=2, label=f'MAE: {valid_errors.mean():.2f} m')
    
    plt.title(f'Esp. {exp_name}: Distribuzione Errori Assoluti')
    plt.xlabel('Errore Assoluto (|Pred - GT|) in metri')
    plt.ylabel('Frequenza')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "error_histogram.png")
    plt.close()
    print(f"  ✓ Grafico Istogramma Errori salvato.")

def create_annotated_video(df_results, input_video_path, depth_dir, output_video_path):
    """Crea un video affiancato con frame RGB, mappa di profondità colorata e annotazioni."""
    
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"ERRORE: Impossibile aprire il video sorgente {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Il video di output avrà il doppio della larghezza (RGB + Depth)
    VIDEO_OUTPUT = str(output_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (frame_width * 2, frame_height))
    
    frame_idx = 0
    pbar = tqdm(total=len(df_results), desc="  Scrittura Video Annotato")

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df_results):
            break
        
        row = df_results.iloc[frame_idx]
        frame_name = row['frame_name']
        
        # Carica la mappa di profondità normalizzata
        normalized_depth_path = depth_dir / frame_name.replace('.png', '_depth_normalized.npy')
        
        if normalized_depth_path.exists():
            depth_normalized = np.load(normalized_depth_path)
            
            # 1. Ridimensiona la mappa di profondità
            if depth_normalized.shape[0] != frame_height or depth_normalized.shape[1] != frame_width:
                 depth_resized = cv2.resize(depth_normalized, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            else:
                 depth_resized = depth_normalized
                 
            # 2. Colorizza la mappa di profondità
            depth_colored = apply_colormap(depth_resized)
            
            # 3. Annotazioni sul frame RGB e sulla mappa di profondità
            #T_px = (int(row['x_pixel']), int(row['y_pixel']))
            # riga 105 (modificata): Carica i valori come float (possono essere NaN)
            T_px = (row['x_pixel'], row['y_pixel'])
            # Disegna la croce sul Target
            #if pd.notna(T_px[0]):
            #    cv2.drawMarker(frame, T_px, (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
            #    cv2.drawMarker(depth_colored, T_px, (255, 255, 255), cv2.MARKER_CROSS, 20, 3)
            # Disegna la croce sul Target
            # riga 108 (modificata): Controlla se ENTRAMBI i valori sono validi (non NaN)
            if pd.notna(T_px[0]) and pd.notna(T_px[1]):
                # riga 109: Conversione a intero solo quando è sicuro
                T_px_int = (int(T_px[0]), int(T_px[1])) 
                # riga 110: Usa T_px_int per disegnare
                cv2.drawMarker(frame, T_px_int, (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
                # riga 111: Usa T_px_int per disegnare
                cv2.drawMarker(depth_colored, T_px_int, (255, 255, 255), cv2.MARKER_CROSS, 20, 3)
    
            # Annotazioni testuali
            text_pred = f"Pred: {row['distance_predicted']:.2f} m" if pd.notna(row['distance_predicted']) else "Pred: N/A"
            text_gt = f"GT: {row['distance_groundtruth']:.2f} m" if pd.notna(row['distance_groundtruth']) else "GT: N/A"
            text_err = f"Error: {row['error']:.2f} m" if pd.notna(row['error']) else "Error: N/A"
            
            cv2.putText(frame, text_pred, (frame_width - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text_gt, (frame_width - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(depth_colored, text_err, (frame_width - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Combina i frame affiancati
            combined = np.hstack([frame, depth_colored])
            
        else:
            # Se la mappa di profondità manca
            combined = np.hstack([frame, frame])
            cv2.putText(combined, "DEPTH MAP MISSING", (frame_width + 50, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        
        out.write(combined)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"\n  ✓ Video Annotato salvato: {output_video_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Analysis and Reporting Script for MDE Pipeline.")
    parser.add_argument('exp_id', type=str, choices=['1', '2', '3', '4'], 
                        help="ID dell'esperimento i cui risultati devono essere analizzati.")
    
    args = parser.parse_args()
    
    EXPERIMENT_DICT = {
        "1": dict(name="exp1_152844"),
        "2": dict(name="exp2_154149"),
        "3": dict(name="exp3_161904"),
        "4": dict(name="exp4_162533"),
    }
    EXP = EXPERIMENT_DICT[args.exp_id]
    EXP_NAME = EXP['name']
    
    # 1. Definizione dei percorsi
    RESULTS_DIR = DATA_ROOT / "data" / "results_dav2" / f"results_{EXP_NAME}"
    INPUT_CSV = RESULTS_DIR / f"results_{EXP_NAME}.csv"
    
    # Percorsi per l'input video e la cache della profondità
    VIDEO_1FPS_PATH = DATA_ROOT / "data" / "preprocessed" / f"video_1fps_{EXP_NAME}.mp4"
    DEPTH_CACHE_DIR = DATA_ROOT / "data" / "depth_map_dav2" / f"relative_{EXP_NAME}"
    OUTPUT_VIDEO = RESULTS_DIR / f"video_annotated_{EXP_NAME}.mp4"
    
    print(f"--- Analysis for Experiment {args.exp_id} ({EXP_NAME}) ---")
    
    if not INPUT_CSV.exists():
        print(f"ERRORE: File risultati non trovato in {INPUT_CSV}")
        print("Assicurarsi di aver eseguito 'main_pipeline.py' per questo esperimento.")
        return

    # 2. Caricamento Dati
    df_results = pd.read_csv(INPUT_CSV)
    
    # 3. Generazione Grafici
    print("\n[STEP 1/2] Generazione Grafici...")
    generate_plots(df_results, RESULTS_DIR, EXP_NAME)
    
    # 4. Creazione Video Annotato
    print("\n[STEP 2/2] Creazione Video Annotato...")
    create_annotated_video(df_results, VIDEO_1FPS_PATH, DEPTH_CACHE_DIR, OUTPUT_VIDEO)
    
    print("\n" + "=" * 60)
    print("Analisi e Reporting completati.")
    print("=" * 60)

if __name__ == "__main__":
    main()