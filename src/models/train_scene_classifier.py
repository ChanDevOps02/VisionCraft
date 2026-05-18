from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


class ImageFolderWithObjectFeatures(torch.utils.data.Dataset):
    def __init__(self, image_folder: datasets.ImageFolder, data_root: Path, feature_path: Path):
        self.image_folder = image_folder
        self.data_root = data_root
        feature_data = np.load(feature_path, allow_pickle=True)
        paths = feature_data["paths"]
        features = feature_data["features"].astype(np.float32)

        self.path_to_feature = {
            str(path): features[idx]
            for idx, path in enumerate(paths.tolist())
        }
        self.feature_dim = int(features.shape[1]) if features.ndim == 2 else 0
        self.classes = image_folder.classes
        self.samples = image_folder.samples

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, index):
        image, label = self.image_folder[index]
        image_path = Path(self.image_folder.samples[index][0])
        relative_path = str(image_path.relative_to(self.data_root))
        object_feature = self.path_to_feature.get(relative_path)
        if object_feature is None:
            object_feature = np.zeros(self.feature_dim, dtype=np.float32)
        return image, label, torch.from_numpy(object_feature)


def build_loaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    object_features_path: str = "",
):
    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(data_root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_root / "val", transform=eval_transform)

    if object_features_path:
        feature_path = Path(object_features_path)
        train_dataset = ImageFolderWithObjectFeatures(train_dataset, data_root, feature_path)
        val_dataset = ImageFolderWithObjectFeatures(val_dataset, data_root, feature_path)

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


class LateFusionSceneClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, object_feature_dim: int):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.object_projection = nn.Sequential(
            nn.Linear(object_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, images, object_features):
        visual_features = self.backbone(images)
        object_features = self.object_projection(object_features)
        fused = torch.cat([visual_features, object_features], dim=1)
        return self.classifier(fused)


def build_model(
    num_classes: int,
    backbone: str = "resnet18",
    fusion_mode: str = "visual-only",
    object_feature_dim: int = 0,
):
    if fusion_mode == "late-fusion":
        return LateFusionSceneClassifier(backbone, num_classes, object_feature_dim)

    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable: bool):
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = trainable
        return

    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def apply_mixup(images, labels, alpha: float, device, object_features=None):
    if alpha <= 0:
        return images, labels, labels, 1.0, object_features

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(images.size(0), device=device)
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]
    mixed_object_features = object_features
    if object_features is not None:
        mixed_object_features = lam * object_features + (1 - lam) * object_features[index]
    return mixed_images, labels_a, labels_b, lam, mixed_object_features


def forward_model(model, images, object_features=None):
    if object_features is not None and hasattr(model, "object_projection"):
        return model(images, object_features)
    return model(images)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, labels, object_features = batch
                object_features = object_features.to(device)
            else:
                images, labels = batch
                object_features = None

            images = images.to(device)
            labels = labels.to(device)
            logits = forward_model(model, images, object_features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def train(args):
    device = get_device()
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")

    train_dataset, _, train_loader, val_loader = build_loaders(
        data_root,
        args.image_size,
        args.batch_size,
        args.num_workers,
        args.object_features_path,
    )
    object_feature_dim = getattr(train_dataset, "feature_dim", 0)
    model = build_model(
        num_classes=len(train_dataset.classes),
        backbone=args.backbone,
        fusion_mode=args.fusion_mode,
        object_feature_dim=object_feature_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_reduce_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    backbone_frozen = False
    for epoch in range(args.epochs):
        should_freeze_backbone = epoch < args.freeze_backbone_epochs
        if should_freeze_backbone and not backbone_frozen:
            set_backbone_trainable(model, trainable=False)
            backbone_frozen = True
            print(f"froze backbone for epoch {epoch + 1}")
        elif not should_freeze_backbone and backbone_frozen:
            set_backbone_trainable(model, trainable=True)
            backbone_frozen = False
            print(f"unfroze backbone at epoch {epoch + 1}")

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for batch in train_loader:
            if len(batch) == 3:
                images, labels, object_features = batch
                object_features = object_features.to(device)
            else:
                images, labels = batch
                object_features = None

            images = images.to(device)
            labels = labels.to(device)
            images, labels_a, labels_b, lam, object_features = apply_mixup(
                images,
                labels,
                args.mixup_alpha,
                device,
                object_features,
            )

            optimizer.zero_grad()
            logits = forward_model(model, images, object_features)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            running_correct += (
                lam * (predictions == labels_a).sum().item()
                + (1 - lam) * (predictions == labels_b).sum().item()
            )
            running_samples += images.size(0)

        train_loss = running_loss / max(running_samples, 1)
        train_acc = running_correct / max(running_samples, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch={epoch + 1} "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "image_size": args.image_size,
                    "backbone": args.backbone,
                    "fusion_mode": args.fusion_mode,
                    "object_feature_dim": object_feature_dim,
                },
                output_path,
            )
            print(f"saved best checkpoint to {output_path}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"early stopping triggered at epoch {epoch + 1} "
                    f"(best epoch: {best_epoch}, best val_acc: {best_val_acc:.4f})"
                )
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Train a scene classifier for VisionCraft.")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root with train/ and val/ folders.")
    parser.add_argument("--output", type=str, default="checkpoints/scene_classifier_resnet18.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--fusion-mode", choices=["visual-only", "late-fusion"], default="visual-only")
    parser.add_argument("--object-features-path", type=str, default="")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
