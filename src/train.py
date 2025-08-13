import argparse
import os
import random
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from augmentation import get_transforms
from model import get_model, save_checkpoint


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_sum += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate_with_neither(model, loader, device, class_names, threshold: float):
    model.eval()
    softmax = nn.Softmax(dim=1)

    total = 0
    confident_correct = 0
    neither_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = softmax(logits)
        confs, preds = probs.max(dim=1)

        for i in range(labels.size(0)):
            total += 1
            if confs[i].item() < threshold:
                neither_count += 1
                continue
            if preds[i].item() == labels[i].item():
                confident_correct += 1

    confident_acc = confident_correct / max((total - neither_count), 1)
    neither_rate = neither_count / max(total, 1)

    return {
        "total_samples": total,
        "neither_count": neither_count,
        "neither_rate": round(neither_rate, 4),
        "confident_accuracy": round(confident_acc, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Train custom CNN for acne/rosacea vs eczema.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--out", type=str, default="model.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--val_dir", type=str, default="val")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tfms, test_tfms = get_transforms(image_size=args.image_size)

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    class_names = train_ds.classes
    print(f"Classes: {class_names}")

    model = get_model(num_classes=len(class_names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    start_time = time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.out, model, class_names)
            print(f"  ↳ Saved best model to {args.out}")

    total_time = time() - start_time
    print(f"Training complete in {total_time/60:.1f} min. Best val_acc: {best_val_acc:.4f}")

    print("\nEvaluating on test set with 'neither' threshold...")
    summary = evaluate_with_neither(model, test_loader, device, class_names, args.threshold)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
