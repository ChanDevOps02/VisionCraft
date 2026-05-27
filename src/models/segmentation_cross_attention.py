from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SegmentationTokenEncoder(nn.Module):
    """Project segmentation ratio vectors into a small token set."""

    def __init__(
        self,
        input_dim: int,
        token_dim: int = 256,
        num_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.projection = nn.Sequential(
            nn.Linear(input_dim, token_dim * 2),
            nn.LayerNorm(token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim * num_tokens),
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, segmentation_features: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(segmentation_features)
        tokens = tokens.view(segmentation_features.size(0), self.num_tokens, self.token_dim)
        return self.output_dropout(self.output_norm(tokens))


class VisualSegmentationCrossAttention(nn.Module):
    """Single-direction cross-attention where visual tokens attend to segmentation tokens."""

    def __init__(
        self,
        visual_dim: int,
        segmentation_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_projection = nn.Linear(visual_dim, segmentation_dim)
        self.visual_norm = nn.LayerNorm(segmentation_dim)
        self.segmentation_norm = nn.LayerNorm(segmentation_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=segmentation_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(segmentation_dim, segmentation_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(segmentation_dim * 4, segmentation_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(segmentation_dim)
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, visual_tokens: torch.Tensor, segmentation_tokens: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.visual_projection(visual_tokens)
        query = self.visual_norm(visual_tokens)
        key_value = self.segmentation_norm(segmentation_tokens)

        attended, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention_weights = attn_weights.detach()

        fused = visual_tokens + attended
        fused = fused + self.ffn(self.ffn_norm(fused))
        return fused


class ResNetSegmentationCrossAttentionSceneClassifier(nn.Module):
    """Scene classifier using ResNet visual tokens and segmentation-ratio cross-attention."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        segmentation_feature_dim: int,
        hidden_dim: int = 256,
        num_segmentation_tokens: int = 4,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.hidden_dim = hidden_dim

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
        self.segmentation_encoder = SegmentationTokenEncoder(
            input_dim=segmentation_feature_dim,
            token_dim=hidden_dim,
            num_tokens=num_segmentation_tokens,
            dropout=dropout,
        )
        self.cross_attention = VisualSegmentationCrossAttention(
            visual_dim=conv_dim,
            segmentation_dim=hidden_dim,
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

    def forward(self, images: torch.Tensor, segmentation_features: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.encode_visual_tokens(images)
        segmentation_tokens = self.segmentation_encoder(segmentation_features)
        fused_visual = self.cross_attention(visual_tokens, segmentation_tokens)
        pooled = self.output_norm(fused_visual).mean(dim=1)
        return self.classifier(pooled)
