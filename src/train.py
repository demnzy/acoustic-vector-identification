import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_datasets
from model import VectorNet
from features import SPECIES_CLASSES


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train baseline acoustic classifier")
    parser.add_argument("--data", type=str, default="data/processed/features.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-path", type=str, default="models/baseline_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, test_ds, _ = load_datasets(
        data_path=args.data,
        seed=args.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    in_features = train_ds.X.shape[1]
    num_classes = len(SPECIES_CLASSES)

    model = VectorNet(in_features=in_features, num_classes=num_classes).to(device)
    print(f"Model initialized: {model.num_parameters():,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print("\n--- Training Baseline Model ---")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "in_features": in_features,
                    "num_classes": num_classes,
                },
                args.save_path,
            )
            saved_tag = " [Best Checkpoint Saved]"
        else:
            saved_tag = ""

        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.1f}% "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}%{saved_tag}"
            )

    # Save class mapping
    label_map = {idx: sp for idx, sp in enumerate(SPECIES_CLASSES)}
    with open("models/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Final evaluation on holdout test set
    checkpoint = torch.load(args.save_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("\n--- Training Complete ---")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Holdout Test Accuracy:    {test_acc * 100:.2f}%")
    print(f"Model saved -> {args.save_path}")


if __name__ == "__main__":
    main()
