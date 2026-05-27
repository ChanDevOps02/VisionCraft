from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_scene_classifier import get_device


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_loaders(data_root: Path, image_size: int, batch_size: int, num_workers: int):
    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(data_root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_root / "val", transform=eval_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_model(num_classes: int, backbone: str):
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CompCars make/model classifier.")
    parser.add_argument("--data-root", type=str, required=True, help="Directory with train/ and val/ folders.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--task-name", choices=["make", "model"], required=True)
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    train_dataset, val_dataset, train_loader, val_loader = build_loaders(
        data_root,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    model = build_model(num_classes=len(train_dataset.classes), backbone=args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        if epoch < args.freeze_backbone_epochs:
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith("fc.")
        elif epoch == args.freeze_backbone_epochs:
            for param in model.parameters():
                param.requires_grad = True

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_samples += images.size(0)

        train_loss = running_loss / max(running_samples, 1)
        train_acc = running_correct / max(running_samples, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "task_name": args.task_name,
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                    "classes": train_dataset.classes,
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                },
                output_path,
            )
            print(f"saved best checkpoint to {output_path}")

    summary = {
        "task_name": args.task_name,
        "backbone": args.backbone,
        "num_classes": len(train_dataset.classes),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
