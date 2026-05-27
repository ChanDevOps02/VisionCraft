from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision import models

from src.models.object_features import COCO80_INDEX


@dataclass
class ObjectTokenBatch:
    class_ids: torch.Tensor
    geometry: torch.Tensor
    valid_mask: torch.Tensor


def detections_to_padded_object_tokens(
    detections: list[dict[str, Any]],
    max_objects: int = 16,
) -> ObjectTokenBatch:
    """Convert YOLO detection dicts into padded token inputs.

    Geometry features per object:
    - confidence
    - bbox center x, center y
    - bbox width, bbox height
    - area ratio
    - thirds distance

    All values are expected to be normalized except confidence and thirds distance,
    which already arrive in [0, 1]-like ranges from our detector pipeline.
    """

    class_ids = torch.zeros(max_objects, dtype=torch.long)
    geometry = torch.zeros(max_objects, 7, dtype=torch.float32)
    valid_mask = torch.zeros(max_objects, dtype=torch.bool)

    for idx, detection in enumerate(detections[:max_objects]):
        label = detection.get("label", "")
        if label not in COCO80_INDEX:
            continue

        x1, y1, x2, y2 = detection.get("bbox", (0, 0, 0, 0))
        class_ids[idx] = COCO80_INDEX[label]
        geometry[idx] = torch.tensor(
            [
                float(detection.get("confidence", 0.0)),
                float((x1 + x2) / 2.0),
                float((y1 + y2) / 2.0),
                float(max(x2 - x1, 0)),
                float(max(y2 - y1, 0)),
                float(detection.get("area_ratio", 0.0)),
                float(detection.get("thirds_distance", 0.0)),
            ],
            dtype=torch.float32,
        )
        valid_mask[idx] = True

    return ObjectTokenBatch(class_ids=class_ids, geometry=geometry, valid_mask=valid_mask)


class ObjectTokenEncoder(nn.Module):
    """Encode YOLO detections as learnable object tokens.

    Each token combines:
    - a class embedding for the COCO category
    - a geometric feature embedding for confidence / bbox / area / thirds score
    """

    def __init__(
        self,
        num_object_classes: int = 80,
        token_dim: int = 256,
        class_embed_dim: int = 96,
        geometry_dim: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.class_embedding = nn.Embedding(num_object_classes, class_embed_dim)
        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_projection = nn.Sequential(
            nn.Linear(class_embed_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.null_token = nn.Parameter(torch.zeros(1, 1, token_dim))

    def forward(
        self,
        class_ids: torch.Tensor,
        geometry: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_feat = self.class_projection(self.class_embedding(class_ids))
        geom_feat = self.geometry_projection(geometry)
        object_tokens = self.output_projection(torch.cat([class_feat, geom_feat], dim=-1))

        # Ensure every sample has at least one valid key/value token.
        no_valid = ~valid_mask.any(dim=1)
        if no_valid.any():
            object_tokens = object_tokens.clone()
            valid_mask = valid_mask.clone()
            object_tokens[no_valid, 0:1, :] = self.null_token.expand(no_valid.sum(), -1, -1)
            valid_mask[no_valid, 0] = True

        return object_tokens, valid_mask


class VisualObjectCrossAttention(nn.Module):
    """Single-direction cross-attention where visual tokens attend to object tokens."""

    def __init__(
        self,
        visual_dim: int,
        object_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_projection = nn.Linear(visual_dim, object_dim)
        self.visual_norm = nn.LayerNorm(object_dim)
        self.object_norm = nn.LayerNorm(object_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=object_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(object_dim, object_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(object_dim * 4, object_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(object_dim)
        self.last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        visual_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        object_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        visual_tokens = self.visual_projection(visual_tokens)
        query = self.visual_norm(visual_tokens)
        key_value = self.object_norm(object_tokens)
        key_padding_mask = ~object_valid_mask

        attended, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention_weights = attn_weights.detach()

        fused = visual_tokens + attended
        fused = fused + self.ffn(self.ffn_norm(fused))
        return fused


class ResNetObjectCrossAttentionSceneClassifier(nn.Module):
    """Scene classifier using ResNet feature-map tokens and YOLO object-token cross-attention."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        hidden_dim: int = 256,
        max_objects: int = 16,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.dropout_rate = dropout

        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            conv_dim = 2048
        else:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            conv_dim = 512

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.object_encoder = ObjectTokenEncoder(token_dim=hidden_dim, dropout=dropout)
        self.cross_attention = VisualObjectCrossAttention(
            visual_dim=conv_dim,
            object_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(images)
        batch_size, channels, height, width = feature_map.shape
        return feature_map.view(batch_size, channels, height * width).transpose(1, 2)

    def forward(
        self,
        images: torch.Tensor,
        object_class_ids: torch.Tensor,
        object_geometry: torch.Tensor,
        object_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        visual_tokens = self.encode_visual_tokens(images)
        object_tokens, object_valid_mask = self.object_encoder(
            object_class_ids,
            object_geometry,
            object_valid_mask,
        )
        fused_visual = self.cross_attention(visual_tokens, object_tokens, object_valid_mask)
        pooled = self.output_norm(fused_visual).mean(dim=1)
        return self.classifier(pooled)
