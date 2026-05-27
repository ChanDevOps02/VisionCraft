from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class VisualTextCrossAttention(nn.Module):
    """Single-direction cross-attention where visual tokens attend to scene text tokens."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.visual_projection(visual_tokens)
        text_tokens = self.text_projection(text_tokens)

        query = self.visual_norm(visual_tokens)
        key_value = self.text_norm(text_tokens)

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


class ResNetTextCrossAttentionSceneClassifier(nn.Module):
    """Scene classifier using ResNet visual tokens and fixed scene-text prototype tokens."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        scene_text_embeddings: torch.Tensor,
        hidden_dim: int = 256,
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

        text_embeddings = scene_text_embeddings.detach().clone().float()
        self.register_buffer("scene_text_embeddings", text_embeddings)
        self.text_embedding_dim = int(text_embeddings.shape[1])

        self.cross_attention = VisualTextCrossAttention(
            visual_dim=conv_dim,
            text_dim=self.text_embedding_dim,
            hidden_dim=hidden_dim,
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

    def get_text_tokens(self, batch_size: int) -> torch.Tensor:
        return self.scene_text_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

    def get_projected_text_tokens(self) -> torch.Tensor:
        text_tokens = self.scene_text_embeddings.unsqueeze(0)
        projected = self.cross_attention.text_projection(text_tokens)
        return self.cross_attention.text_norm(projected).squeeze(0)

    def extract_fused_latent(self, images: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.encode_visual_tokens(images)
        text_tokens = self.get_text_tokens(images.size(0))
        fused_visual = self.cross_attention(visual_tokens, text_tokens)
        return self.output_norm(fused_visual).mean(dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = self.extract_fused_latent(images)
        return self.classifier(pooled)
