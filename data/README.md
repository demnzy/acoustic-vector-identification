# Dataset Documentation

This repository uses acoustic flight recordings from the public **Wingbeats** dataset (Fanioudakis et al. / Kaggle), sampled at 8 kHz mono (0.625 seconds per recording).

## Extracted Feature Set

Features in `data/processed/` are derived from balanced sampling across 6 primary disease vector species:
- *Aedes aegypti*
- *Aedes albopictus*
- *Anopheles arabiensis*
- *Anopheles gambiae*
- *Culex pipiens*
- *Culex quinquefasciatus*

Each sample contains a 48-dimensional feature vector:
1. `mfcc_mean_0` to `mfcc_mean_12`: Mean of 13 Mel-Frequency Cepstral Coefficients across temporal frames.
2. `mfcc_std_0` to `mfcc_std_12`: Standard deviation of 13 MFCCs.
3. `delta_mfcc_mean_0` to `delta_mfcc_mean_12`: Mean of first-order temporal derivatives (velocity).
4. `centroid_mean`, `centroid_std`, `centroid_median`, `centroid_min`, `centroid_max`: Spectral centroid distribution capturing fundamental wingbeat harmonics.
5. `rolloff_mean`, `rolloff_std`: 85% spectral energy threshold.
6. `zcr_mean`, `zcr_std`: Zero-crossing rate statistics.

## Regeneration

To regenerate the processed features directly from the raw archive:

```bash
python src/features.py --archive "path/to/archive (3).zip" --samples-per-class 400
```
