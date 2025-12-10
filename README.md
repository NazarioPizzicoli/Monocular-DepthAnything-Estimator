# RealTime Monocular Depth Estimation with Metric Calibration

Una pipeline completa e modulare (Python/PyTorch) che trasforma la stima della profondità relativa (Relative Depth Estimation - RDE) di Depth Anything V2 in accurate misurazioni di profondità metrica (Metric Depth Estimation - MDE) in tempo reale, utilizzando la calibrazione con Ground Control Points (GCPs).

## Punti di Forza e Innovazione

Questo progetto affronta il problema centrale della Monocular Depth Estimation: la **perdita di scala metrica**. Dimostra una soluzione ingegneristica robusta per applicazioni critiche come la robotica o i sistemi di navigazione marittima.

### Valore Scientifico e Tecnologico

* **Metricizzazione Zero-Shot:** Sfrutta il potere di generalizzazione di **Depth Anything V2** (basato su architettura Transformer DINOv2) e lo eleva alla metrologia.
* **Calibrazione Robusta:** Implementa una tecnica di calibrazione *per frame* basata su Punti di Controllo a Terra (GCPs, nel nostro caso boe $\mathbf{B}_{i}$) per scalare la profondità relativa nel dominio metrico (metri). 
* **Sincronizzazione Critica:** Risolve i problemi di *data fusion* e *time synchronization* abbinando con precisione frame video (CV2) con metadati di sensori esterni (GPS, file MCAP ROS2) e annotazioni pixel (YOLO) con tolleranza temporale definita.

### Architettura e Ingegneria del Software

* **Architettura Modulare (Pipeline a 3 Fasi):** Il progetto è diviso in script indipendenti (Data Preparation, Inference/Calibration, Analysis), che lo rende testabile, manutenibile e scalabile.
* **Pipeline Real-Time Ready:** Progettato per operare su flussi video a bassa latenza (1 FPS di sampling), con cache intelligente delle mappe di profondità (`--reload`) per ottimizzare l'uso della GPU.
* **Gestione Dati Strutturata:** Utilizza un'organizzazione `data/` chiara (`raw/`, `preprocessed/`, `results/`) che garantisce la riproducibilità di ogni esperimento.

---

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* GPU con supporto CUDA (Consigliata per inferenza V2 Large/Base)

### Setting up the Environment
```bash
git clone [https://github.com/NazarioPizzicoli/Monocular-DepthAnything-Estimator.git](https://github.com/NazarioPizzicoli/Monocular-DepthAnything-Estimator.git)
cd MDE-GitHub
python -m venv venv
source venv/bin/activate
```

## Installation Dependencies
```
pip install -r requirements.txt
```

**Nota**: Potrebbe essere necessario scaricare manualmente i pesi del modello Depth Anything V2 e posizionarli in models/Depth-Anything-V2/checkpoints/.

## Usage
La pipeline è eseguita in tre fasi sequenziali. Sostituisci <EXP_ID> con l'ID dell'esperimento (e.g., '1', '2', '3', '4').

### Fase 1:
Data Preprocessing & Matching (scripts/data_preparation/preprocess_and_match.py)
**Scopo**: Sincronizzazione temporale, estrazione frame (1 FPS) e abbinamento di metadati GPS/YOLO.

```
# Eseguire la preparazione dei dati per l'esperimento 1
python scripts/data_preparation/preprocess_and_match.py --exp 1
```

**Output**: I file necessari (gps_exp1_152844.csv, anchor_points.csv, video_1fps_exp1_152844.mp4) vengono salvati in data/preprocessed/.

### Fase 2:
MDE Inference & Metric Calibration (main_pipeline.py)

**Scopo**: Esecuzione del modello Depth Anything V2, estrazione delle profondità relative di Target e GCPs, e calibrazione metrica con il metodo Least Squares sul modello inverso ($1/D_{metric} = a \cdot D_{relative} + b$).

```
# Eseguire l'inferenza e la calibrazione per l'esperimento 1
python main_pipeline.py 1

# Opzionale: saltare l'inferenza MDE se le mappe sono già in cache
# python main_pipeline.py 1 --reload
```

**Output**: Il file di risultati results_exp1_152844.csv (che include Predizione, GT ed Errore) viene salvato in data/results_dav2/results_exp1_152844/.

### Fase 3:
Analysis & Reporting (scripts/analysis/generate_plots.py)

**Scopo**: Generazione di grafici di performance (serie temporale di profondità, distribuzione degli errori) e creazione del video finale annotato che affianca RGB e MDE metrico.

```
# Eseguire l'analisi e il reporting per l'esperimento 1
python scripts/analysis/generate_plots.py 1
```

**Output**: Grafici e il video video_annotated_exp1_152844.mp4 nella cartella dei risultati.

## Contribuire
Sentiti libero di aprire issue per bug, suggerimenti o nuove funzionalità. Pull requests sono benvenute, specialmente per:
- Miglioramenti alla funzione di calibrazione metrica (e.g., RANSAC per outlier).
- Integrazione di modelli MDE alternativi (e.g., Marigold).
- Miglioramenti alla visualizzazione 3D (Top-View GPS/Depth).