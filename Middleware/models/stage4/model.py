"""
Model architectures with transfer learning and regularization.
Includes Focal Loss for imbalanced classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """Compute focal loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()


class ComponentClassifier(nn.Module):
    """
    Transfer learning classifier with pretrained backbone.
    Supports EfficientNet, ResNet, and ViT architectures.
    """
    
    def __init__(self, backbone='efficientnet_b3', num_classes=22, 
                 dropout_rate=0.4, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Load backbone
        if backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = 1536
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'vit_base':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            feature_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove classification head
        if backbone == 'vit_base':
            self.feature_extractor = nn.Sequential(
                *list(self.backbone.children())[:-1]
            )
        else:
            self.feature_extractor = nn.Sequential(
                *list(self.backbone.children())[:-1]
            )
        
        # New classification head with regularization
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if backbone != 'vit_base' else nn.Identity(),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        features = self.feature_extractor(x)
        if self.backbone_name == 'vit_base':
            features = features[:, 1:]
            features = features.mean(dim=1)
        logits = self.head(features)
        return logits
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
