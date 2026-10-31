# Acoustic Vector Identification: Feature Pipeline & Baseline Model

Undergraduate research exploring acoustic identification of disease vectors from wingbeat frequency spectra to support future targeted monitoring and localized control.

- **Supervised by:** Dr. J. A. Bala
- **Author:** Oluwatobiloba Davies
- **Framework:** PyTorch & Signal Processing (`librosa`, `scipy`, `torchaudio`)
- **Status:** Proof-of-Concept Baseline (Aug. 2026 – Present)

---

## Overview

Existing mosquito vector control methods rely heavily on passive traps or broad, environmentally harmful chemical fogging. Optical and computer vision systems consistently struggle because disease vectors are tiny, fast-moving, and primarily active in darkness.

Acoustic identification bypasses visual constraints by capturing species-specific wingbeat frequency signatures. This repository establishes:
1. An automated acoustic feature-extraction pipeline combining **Mel-Frequency Cepstral Coefficients (MFCC)** and **spectral centroid statistics**.
2. A lightweight **PyTorch baseline classifier (`VectorNet`)** designed for low-power edge microcontrollers and field deployment.

---

## Dataset

We evaluate on acoustic wingbeat recordings from the public **Wingbeats** dataset (Fanioudakis et al. / Kaggle), recorded at 8,000 Hz mono (~0.625 seconds per recording).

Target species include 6 primary disease vectors:
- `Ae. aegypti` (Dengue, Zika, Yellow Fever)
- `Ae. albopictus` (Chikungunya, Dengue)
- `An. arabiensis` (Malaria)
- `An. gambiae` (Major African malaria vector)
- `C. pipiens` (West Nile Virus)
- `C. quinquefasciatus` (Lymphatic filariasis)

A curated, balanced sample of 2,400 recordings (400 per class) is pre-extracted in `data/processed/` for direct reproducibility.

---

## Feature Extraction Pipeline

The pipeline extracts a 48-dimensional representation per audio recording:

| Feature Group | Dim | Description |
|---|---|---|
| **MFCC Means** | 13 | Mean of 13 Mel-Frequency Cepstral Coefficients across time frames |
| **MFCC Stds** | 13 | Temporal variation / variance of cepstral envelope |
| **Delta MFCC Means** | 13 | First-order temporal derivatives (velocity) |
| **Spectral Centroid Stats** | 5 | Mean, standard deviation, median, min, max (captures fundamental wingbeat harmonic distribution) |
| **Spectral Rolloff** | 2 | 85% energy frequency distribution (mean, std) |
| **Zero Crossing Rate** | 2 | Mean and std of signal sign changes |

---

## Model Architecture (`VectorNet`)

To meet edge hardware constraints (e.g., STM32, ESP32, Raspberry Pi), the baseline model is intentionally kept lightweight:

```
Input (48 features)
  ├── Linear(48 -> 128) + BatchNorm1d + ReLU + Dropout(0.2)
  ├── Linear(128 -> 64)  + BatchNorm1d + ReLU + Dropout(0.2)
  └── Linear(64 -> 6)    (Logits over 6 mosquito species)
```

- **Trainable Parameters:** 15,302
- **Model Checkpoint Size:** ~68 KB float32 (fits within MCU SRAM/Flash)
- **Inference Latency:** < 1 ms on standard CPU

---

## Baseline Results

Evaluated on a held-out test split (15% stratified test set, 360 audio samples):

```
                     precision    recall  f1-score   support

        Ae. aegypti     0.6923    0.6000    0.6429        60
     Ae. albopictus     0.7538    0.8167    0.7840        60
     An. arabiensis     0.6481    0.5833    0.6140        60
        An. gambiae     0.6923    0.7500    0.7200        60
         C. pipiens     0.7463    0.8333    0.7874        60
C. quinquefasciatus     0.8947    0.8500    0.8718        60

           accuracy                         0.7389       360
          macro avg     0.7379    0.7389    0.7367       360
       weighted avg     0.7379    0.7389    0.7367       360
```

- **Baseline Test Accuracy:** **73.89%** across 6 species (vs. 16.67% random baseline).
- **Confusion Matrix:** Saved at `reports/confusion_matrix.png`.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/demnzy/acoustic-vector-identification.git
cd acoustic-vector-identification
pip install -r requirements.txt
```

### 2. Feature Extraction (Optional if using processed data)

To extract features directly from a raw Wingbeats zip archive:

```bash
python src/features.py --archive "path/to/archive (3).zip" --samples-per-class 400
```

### 3. Training the PyTorch Baseline

```bash
python src/train.py --epochs 40 --batch-size 32 --lr 0.001
```

### 4. Evaluation & Reports

```bash
python src/evaluate.py
```

---

## Repository Structure

```
acoustic-vector-identification/
├── data/
│   ├── processed/
│   │   ├── features.csv          # Extracted 48-dim feature table
│   │   └── features.npz          # Compressed tensors for DataLoader
│   └── README.md
├── docs/
│   ├── Meta Analysis Table - Oluwatobiloba Davies.pdf
│   └── VectorSonic AI Problem Statement - Oluwatobiloba Davies.pdf
├── models/
│   ├── baseline_model.pth        # Saved PyTorch checkpoint
│   ├── label_map.json            # Species class index dictionary
│   └── scaler_params.json        # Standalone normalization statistics
├── reports/
│   ├── classification_report.txt # Per-class precision, recall, F1
│   └── confusion_matrix.png      # Confusion matrix plot
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Dataset splits & DataLoader
│   ├── evaluate.py               # Test set evaluation & confusion matrix
│   ├── features.py               # MFCC & spectral centroid extraction pipeline
│   ├── model.py                  # VectorNet PyTorch architecture
│   └── train.py                  # Training loop & checkpointing
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Ongoing & Future Work

- **INT8 Quantization:** Compiling `VectorNet` via PyTorch Mobile / ONNX Runtime / CMSIS-NN for microcontrollers.
- **Continuous Stream Inference:** Sliding window detection on ambient continuous audio streams to differentiate background noise from wingbeat bursts.
- **Closed-Loop Actuation:** Linking acoustic identification output to targeted localized elimination mechanisms (e.g. focused acoustic disruption or low-power targeted lasers).
