# src/model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DeepfakeClassifier(nn.Module):
    def __init__(self, freeze_blocks: int = 5, dropout: float = 0.4, backbone: str = 'b4'):
        super().__init__()

        if backbone == 'b4':
            base = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            feature_dim = 1792
        else:  # b0 for ablation
            base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            feature_dim = 1280

        self.features = base.features
        self.avgpool = base.avgpool

        for i, block in enumerate(self.features):
            if i < freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)  # raw logits — apply sigmoid at inference
