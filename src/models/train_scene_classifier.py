from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.object_cross_attention import ResNetObjectCrossAttentionSceneClassifier
from src.models.segmentation_cross_attention import ResNetSegmentationCrossAttentionSceneClassifier
from src.models.text_cross_attention import ResNetTextCrossAttentionSceneClassifier


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


class ImageFolderWithObjectTokens(torch.utils.data.Dataset):
    def __init__(
        self,
        image_folder: datasets.ImageFolder,
        data_root: Path,
        token_path: Path,
        max_objects_override: int = 0,
    ):
        self.image_folder = image_folder
        self.data_root = data_root
        token_data = np.load(token_path, allow_pickle=True)
        paths = token_data["paths"]
        class_ids = token_data["class_ids"].astype(np.int64)
        geometry = token_data["geometry"].astype(np.float32)
        valid_mask = token_data["valid_mask"].astype(bool)

        if max_objects_override > 0 and class_ids.ndim == 2:
            class_ids = class_ids[:, :max_objects_override]
            geometry = geometry[:, :max_objects_override, :]
            valid_mask = valid_mask[:, :max_objects_override]

        self.path_to_tokens = {
            str(path): (
                class_ids[idx],
                geometry[idx],
                valid_mask[idx],
            )
            for idx, path in enumerate(paths.tolist())
        }
        self.max_objects = int(class_ids.shape[1]) if class_ids.ndim == 2 else 0
        self.classes = image_folder.classes
        self.samples = image_folder.samples

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, index):
        image, label = self.image_folder[index]
        image_path = Path(self.image_folder.samples[index][0])
        relative_path = str(image_path.relative_to(self.data_root))
        token_tuple = self.path_to_tokens.get(relative_path)
        if token_tuple is None:
            class_ids = np.zeros((self.max_objects,), dtype=np.int64)
            geometry = np.zeros((self.max_objects, 7), dtype=np.float32)
            valid_mask = np.zeros((self.max_objects,), dtype=bool)
        else:
            class_ids, geometry, valid_mask = token_tuple
        return (
            image,
            label,
            torch.from_numpy(class_ids),
            torch.from_numpy(geometry),
            torch.from_numpy(valid_mask),
        )


class ImageFolderWithSegmentationFeatures(torch.utils.data.Dataset):
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
        segmentation_feature = self.path_to_feature.get(relative_path)
        if segmentation_feature is None:
            segmentation_feature = np.zeros(self.feature_dim, dtype=np.float32)
        return image, label, torch.from_numpy(segmentation_feature)


def build_loaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    object_features_path: str = "",
    object_tokens_path: str = "",
    segmentation_features_path: str = "",
    fusion_mode: str = "visual-only",
    object_token_limit: int = 0,
):
    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(data_root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_root / "val", transform=eval_transform)

    if fusion_mode == "late-fusion":
        feature_path = Path(object_features_path)
        train_dataset = ImageFolderWithObjectFeatures(train_dataset, data_root, feature_path)
        val_dataset = ImageFolderWithObjectFeatures(val_dataset, data_root, feature_path)
    elif fusion_mode == "cross-attention":
        token_path = Path(object_tokens_path)
        train_dataset = ImageFolderWithObjectTokens(
            train_dataset,
            data_root,
            token_path,
            max_objects_override=object_token_limit,
        )
        val_dataset = ImageFolderWithObjectTokens(
            val_dataset,
            data_root,
            token_path,
            max_objects_override=object_token_limit,
        )
    elif fusion_mode == "segmentation-cross-attention":
        feature_path = Path(segmentation_features_path)
        train_dataset = ImageFolderWithSegmentationFeatures(train_dataset, data_root, feature_path)
        val_dataset = ImageFolderWithSegmentationFeatures(val_dataset, data_root, feature_path)

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


def unpack_batch(batch, device):
    if len(batch) == 5:
        images, labels, object_class_ids, object_geometry, object_valid_mask = batch
        return {
            "images": images.to(device),
            "labels": labels.to(device),
            "object_features": None,
            "segmentation_features": None,
            "object_class_ids": object_class_ids.to(device),
            "object_geometry": object_geometry.to(device),
            "object_valid_mask": object_valid_mask.to(device),
        }

    if len(batch) == 3:
        images, labels, aux_features = batch
        return {
            "images": images.to(device),
            "labels": labels.to(device),
            "object_features": aux_features.to(device),
            "segmentation_features": aux_features.to(device),
            "object_class_ids": None,
            "object_geometry": None,
            "object_valid_mask": None,
        }

    images, labels = batch
    return {
        "images": images.to(device),
        "labels": labels.to(device),
        "object_features": None,
        "segmentation_features": None,
        "object_class_ids": None,
        "object_geometry": None,
        "object_valid_mask": None,
    }


def build_model(
    num_classes: int,
    backbone: str = "resnet18",
    fusion_mode: str = "visual-only",
    object_feature_dim: int = 0,
    object_max_objects: int = 16,
    segmentation_feature_dim: int = 0,
    scene_text_embeddings: torch.Tensor | None = None,
    cross_attention_dropout: float = 0.1,
):
    if fusion_mode == "late-fusion":
        return LateFusionSceneClassifier(backbone, num_classes, object_feature_dim)
    if fusion_mode == "cross-attention":
        return ResNetObjectCrossAttentionSceneClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            max_objects=object_max_objects,
            dropout=cross_attention_dropout,
        )
    if fusion_mode == "segmentation-cross-attention":
        return ResNetSegmentationCrossAttentionSceneClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            segmentation_feature_dim=segmentation_feature_dim,
            dropout=cross_attention_dropout,
        )
    if fusion_mode == "text-cross-attention":
        if scene_text_embeddings is None:
            raise ValueError("text-cross-attention requires scene_text_embeddings")
        return ResNetTextCrossAttentionSceneClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            scene_text_embeddings=scene_text_embeddings,
            dropout=cross_attention_dropout,
        )

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


def apply_mixup(images, labels, alpha: float, device, aux_features=None):
    if alpha <= 0:
        return images, labels, labels, 1.0, aux_features

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(images.size(0), device=device)
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]
    mixed_aux_features = aux_features
    if aux_features is not None:
        mixed_aux_features = lam * aux_features + (1 - lam) * aux_features[index]
    return mixed_images, labels_a, labels_b, lam, mixed_aux_features


def compute_text_contrastive_loss(
    model: ResNetTextCrossAttentionSceneClassifier,
    fused_latent: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    latent_norm = F.normalize(fused_latent, dim=-1)
    text_norm = F.normalize(model.get_projected_text_tokens(), dim=-1)
    similarity = latent_norm @ text_norm.transpose(0, 1)
    similarity = similarity / max(temperature, 1e-6)
    return F.cross_entropy(similarity, labels)


def forward_model(
    model,
    images,
    object_features=None,
    segmentation_features=None,
    object_class_ids=None,
    object_geometry=None,
    object_valid_mask=None,
):
    if (
        object_class_ids is not None
        and object_geometry is not None
        and object_valid_mask is not None
        and isinstance(model, ResNetObjectCrossAttentionSceneClassifier)
    ):
        return model(images, object_class_ids, object_geometry, object_valid_mask)
    if segmentation_features is not None and isinstance(model, ResNetSegmentationCrossAttentionSceneClassifier):
        return model(images, segmentation_features)
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
            batch_data = unpack_batch(batch, device)
            logits = forward_model(
                model,
                batch_data["images"],
                batch_data["object_features"],
                batch_data["segmentation_features"],
                batch_data["object_class_ids"],
                batch_data["object_geometry"],
                batch_data["object_valid_mask"],
            )
            loss = criterion(logits, batch_data["labels"])

            total_loss += loss.item() * batch_data["images"].size(0)
            total_correct += (logits.argmax(dim=1) == batch_data["labels"]).sum().item()
            total_samples += batch_data["images"].size(0)

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
        args.object_tokens_path,
        args.segmentation_features_path,
        args.fusion_mode,
        args.object_token_limit,
    )
    object_feature_dim = getattr(train_dataset, "feature_dim", 0)
    object_max_objects = getattr(train_dataset, "max_objects", 16)
    segmentation_feature_dim = (
        getattr(train_dataset, "feature_dim", 0)
        if args.fusion_mode == "segmentation-cross-attention"
        else 0
    )
    scene_text_embeddings = None
    if args.fusion_mode == "text-cross-attention":
        if not args.scene_text_embeddings_path:
            raise ValueError("text-cross-attention requires --scene-text-embeddings-path")
        text_data = np.load(args.scene_text_embeddings_path, allow_pickle=True)
        scene_text_embeddings = torch.from_numpy(text_data["embeddings"].astype(np.float32))
    model = build_model(
        num_classes=len(train_dataset.classes),
        backbone=args.backbone,
        fusion_mode=args.fusion_mode,
        object_feature_dim=object_feature_dim,
        object_max_objects=object_max_objects,
        segmentation_feature_dim=segmentation_feature_dim,
        scene_text_embeddings=scene_text_embeddings,
        cross_attention_dropout=args.cross_attention_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if hasattr(model, "backbone") and args.backbone_lr > 0:
        backbone_params = list(model.backbone.parameters())
        backbone_param_ids = {id(param) for param in backbone_params}
        head_params = [
            param
            for param in model.parameters()
            if id(param) not in backbone_param_ids
        ]
        param_groups = [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr},
        ]
    else:
        param_groups = model.parameters()

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(param_groups, lr=args.lr)
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
            batch_data = unpack_batch(batch, device)
            images = batch_data["images"]
            labels = batch_data["labels"]
            object_features = batch_data["object_features"]
            segmentation_features = batch_data["segmentation_features"]
            object_class_ids = batch_data["object_class_ids"]
            object_geometry = batch_data["object_geometry"]
            object_valid_mask = batch_data["object_valid_mask"]

            if object_class_ids is None:
                mixup_features = segmentation_features if segmentation_features is not None else object_features
                images, labels_a, labels_b, lam, mixed_features = apply_mixup(
                    images,
                    labels,
                    args.mixup_alpha,
                    device,
                    mixup_features,
                )
                if segmentation_features is not None:
                    segmentation_features = mixed_features
                else:
                    object_features = mixed_features
            else:
                labels_a = labels
                labels_b = labels
                lam = 1.0

            optimizer.zero_grad()
            contrastive_loss = None
            if isinstance(model, ResNetTextCrossAttentionSceneClassifier):
                fused_latent = model.extract_fused_latent(images)
                logits = model.classifier(fused_latent)
                if args.text_contrastive_weight > 0:
                    contrastive_loss = (
                        lam * compute_text_contrastive_loss(
                            model,
                            fused_latent,
                            labels_a,
                            args.text_contrastive_temperature,
                        )
                        + (1 - lam) * compute_text_contrastive_loss(
                            model,
                            fused_latent,
                            labels_b,
                            args.text_contrastive_temperature,
                        )
                    )
            else:
                logits = forward_model(
                    model,
                    images,
                    object_features,
                    segmentation_features,
                    object_class_ids,
                    object_geometry,
                    object_valid_mask,
                )

            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            if contrastive_loss is not None:
                loss = loss + args.text_contrastive_weight * contrastive_loss
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
                    "object_max_objects": object_max_objects,
                    "segmentation_feature_dim": segmentation_feature_dim,
                    "scene_text_embeddings": scene_text_embeddings.cpu() if scene_text_embeddings is not None else None,
                    "cross_attention_dropout": args.cross_attention_dropout,
                    "text_contrastive_weight": args.text_contrastive_weight,
                    "text_contrastive_temperature": args.text_contrastive_temperature,
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
    parser.add_argument(
        "--fusion-mode",
        choices=["visual-only", "late-fusion", "cross-attention", "segmentation-cross-attention", "text-cross-attention"],
        default="visual-only",
    )
    parser.add_argument("--object-features-path", type=str, default="")
    parser.add_argument("--object-tokens-path", type=str, default="")
    parser.add_argument("--segmentation-features-path", type=str, default="")
    parser.add_argument("--scene-text-embeddings-path", type=str, default="")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cross-attention-dropout", type=float, default=0.1)
    parser.add_argument("--text-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--text-contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--object-token-limit", type=int, default=0)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
