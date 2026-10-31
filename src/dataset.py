import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class WingbeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_datasets(
    data_path="data/processed/features.npz",
    val_size=0.15,
    test_size=0.15,
    seed=42,
    scaler_out="models/scaler_params.json",
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found: {data_path}. Run features.py first.")

    if data_path.endswith(".npz"):
        data = np.load(data_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
    elif data_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(data_path)
        feature_cols = [c for c in df.columns if c not in ("species", "label")]
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.int64)
    else:
        raise ValueError("Unsupported format, use .npz or .csv")

    # Split into train + temp (val + test)
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=seed, stratify=y
    )

    # Split temp into val and test
    relative_test = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=seed, stratify=y_temp
    )

    # Fit scaler strictly on train split
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Persist scaler parameters for standalone inference
    if scaler_out:
        os.makedirs(os.path.dirname(scaler_out), exist_ok=True)
        with open(scaler_out, "w") as f:
            json.dump(
                {
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                },
                f,
                indent=2,
            )

    return (
        WingbeatDataset(X_train, y_train),
        WingbeatDataset(X_val, y_val),
        WingbeatDataset(X_test, y_test),
        scaler,
    )
