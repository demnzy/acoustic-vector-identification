import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from dataset import load_datasets
from model import VectorNet
from features import SPECIES_CLASSES


def plot_confusion_matrix(cm, classes, out_path="reports/confusion_matrix.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Mosquito Species Identification Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on holdout test set")
    parser.add_argument("--data", type=str, default="data/processed/features.npz")
    parser.add_argument("--model-path", type=str, default="models/baseline_model.pth")
    parser.add_argument("--report-path", type=str, default="reports/classification_report.txt")
    parser.add_argument("--fig-path", type=str, default="reports/confusion_matrix.png")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load holdout test split
    _, _, test_ds, _ = load_datasets(data_path=args.data)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}. Train the model first.")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model = VectorNet(
        in_features=checkpoint.get("in_features", 48),
        num_classes=checkpoint.get("num_classes", len(SPECIES_CLASSES)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        outputs = model(test_ds.X)
        preds = torch.argmax(outputs, dim=1).numpy()
        targets = test_ds.y.numpy()

    report = classification_report(
        targets,
        preds,
        target_names=SPECIES_CLASSES,
        digits=4,
    )

    print("\n--- Test Set Classification Report ---")
    print(report)

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        f.write("Mosquito Vector Acoustic Classification Report\n")
        f.write("Dataset: Public Wingbeats Audio (8 kHz)\n")
        f.write("Model: VectorNet (Edge-Friendly Baseline MLP)\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Saved report -> {args.report_path}")

    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, SPECIES_CLASSES, out_path=args.fig_path)


if __name__ == "__main__":
    main()
