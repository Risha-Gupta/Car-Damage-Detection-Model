"""
Component classification model inference wrapper.
Supports Vision Transformer and other deep learning models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Any
import logging

log = logging.getLogger(__name__)


class Model:
    """Component classification model wrapper."""
    
    def __init__(self, path: str, arch: str = "vit", 
                 num_classes: int = 22, device: str = "cuda"):
        """
        Initialize classifier.
        
        Args:
            path: Path to model checkpoint
            arch: Model architecture ("vit", "efficientnet", "resnet")
            num_classes: Number of component classes
            device: "cuda" or "cpu"
        """
        self.path = path
        self.arch = arch
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net = self._load()
        self.net.eval()
    
    def _load(self) -> nn.Module:
        """Load model from checkpoint."""
        try:
            log.info(f"Loading {self.arch} from {self.path}")
            
            # Example: Load Vision Transformer
            from torchvision.models import vit_b_16
            net = vit_b_16(pretrained=False)
            net.heads.head = nn.Linear(768, self.num_classes)
            net.to(self.device)
            
            log.info(f"Model loaded on {self.device}")
            return net
        except Exception as e:
            log.error(f"Failed to load model: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict(self, roi: np.ndarray) -> Tuple[List[int], List[float]]:
        """
        Predict component labels for single ROI.
        
        Args:
            roi: Preprocessed ROI image (normalized numpy array)
            
        Returns:
            Tuple of (component indices, confidence scores)
        """
        try:
            # Convert to tensor
            tensor = torch.from_numpy(roi).float().to(self.device)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            out = self.net(tensor)
            probs = torch.softmax(out, dim=1)[0]
            
            # Get top predictions
            k = min(3, self.num_classes)
            scores, indices = torch.topk(probs, k)
            
            return indices.cpu().numpy().tolist(), scores.cpu().numpy().tolist()
        except Exception as e:
            log.error(f"Inference error: {str(e)}")
            return [], []
    
    @torch.no_grad()
    def predict_batch(self, batch: np.ndarray) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Predict component labels for batch of ROIs.
        
        Args:
            batch: Batch of preprocessed ROIs (B, H, W, C)
            
        Returns:
            Tuple of (batch indices, batch scores)
        """
        try:
            tensor = torch.from_numpy(batch).float().to(self.device)
            out = self.net(tensor)
            probs = torch.softmax(out, dim=1)
            
            indices = []
            scores = []
            
            for p in probs:
                k = min(3, self.num_classes)
                top_scores, top_indices = torch.topk(p, k)
                indices.append(top_indices.cpu().numpy().tolist())
                scores.append(top_scores.cpu().numpy().tolist())
            
            return indices, scores
        except Exception as e:
            log.error(f"Batch inference error: {str(e)}")
            return [], []
