import os
import io
import zipfile
import random
import argparse
import numpy as np
import soundfile as sf
import librosa

SPECIES_CLASSES = [
    "Ae. aegypti",
    "Ae. albopictus",
    "An. arabiensis",
    "An. gambiae",
    "C. pipiens",
    "C. quinquefasciatus",
]

DEFAULT_ARCHIVE = os.path.expanduser(r"~\Downloads\archive (3).zip")


def get_feature_names(n_mfcc=13):
    names = []
    names += [f"mfcc_mean_{i}" for i in range(n_mfcc)]
    names += [f"mfcc_std_{i}" for i in range(n_mfcc)]
    names += [f"delta_mfcc_mean_{i}" for i in range(n_mfcc)]
    names += [
        "centroid_mean",
        "centroid_std",
        "centroid_median",
        "centroid_min",
        "centroid_max",
    ]
    names += ["rolloff_mean", "rolloff_std"]
    names += ["zcr_mean", "zcr_std"]
    return names


def extract_audio_features(audio, sr=8000, n_mfcc=13, n_fft=512, hop_length=128):
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    # Mel-Frequency Cepstral Coefficients
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # First-order delta MFCCs
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta_mfcc, axis=1)

    # Spectral centroid statistics
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    sc_stats = [
        np.mean(centroid),
        np.std(centroid),
        np.median(centroid),
        np.min(centroid),
        np.max(centroid),
    ]

    # Spectral rolloff (85% energy frequency)
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
    )[0]
    ro_stats = [np.mean(rolloff), np.std(rolloff)]

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=n_fft, hop_length=hop_length
    )[0]
    zcr_stats = [np.mean(zcr), np.std(zcr)]

    return np.concatenate([mfcc_mean, mfcc_std, delta_mean, sc_stats, ro_stats, zcr_stats])


def extract_from_archive(archive_path, samples_per_class=400, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print(f"Reading dataset archive: {archive_path}")
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    features_list = []
    labels_list = []
    species_names = []

    with zipfile.ZipFile(archive_path, "r") as z:
        all_wavs = [n for n in z.namelist() if n.endswith(".wav")]

        for label_idx, sp in enumerate(SPECIES_CLASSES):
            matching = [n for n in all_wavs if f"/{sp}/" in n]
            if len(matching) == 0:
                print(f"Warning: No samples found for {sp}")
                continue

            k = min(samples_per_class, len(matching))
            selected = random.sample(matching, k)
            print(f"Extracting features for {sp}: {k} samples...")

            for item in selected:
                try:
                    data = z.read(item)
                    audio, sr = sf.read(io.BytesIO(data))
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1)

                    feat = extract_audio_features(audio, sr=sr)
                    features_list.append(feat)
                    labels_list.append(label_idx)
                    species_names.append(sp)
                except Exception as e:
                    print(f"Skipping corrupt file {item}: {e}")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    return X, y, species_names


def save_dataset(X, y, species_names, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    feat_names = get_feature_names()

    # Save compact NPZ for PyTorch DataLoader
    npz_path = os.path.join(output_dir, "features.npz")
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        feature_names=feat_names,
        classes=SPECIES_CLASSES,
    )
    print(f"Saved NPZ dataset -> {npz_path} ({X.shape[0]} samples, {X.shape[1]} features)")

    # Save CSV for transparency and reproducibility
    csv_path = os.path.join(output_dir, "features.csv")
    header = ",".join(feat_names + ["species", "label"])
    rows = []
    for i in range(len(X)):
        row_vals = [f"{v:.6f}" for v in X[i]]
        row_vals.append(species_names[i])
        row_vals.append(str(y[i]))
        rows.append(",".join(row_vals))

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"Saved CSV dataset -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wingbeat acoustic feature extraction pipeline")
    parser.add_argument("--archive", type=str, default=DEFAULT_ARCHIVE, help="Path to Wingbeats zip")
    parser.add_argument("--samples-per-class", type=int, default=400, help="Samples per mosquito species")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    X, y, sp_names = extract_from_archive(
        archive_path=args.archive,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    save_dataset(X, y, sp_names, output_dir=args.output_dir)
