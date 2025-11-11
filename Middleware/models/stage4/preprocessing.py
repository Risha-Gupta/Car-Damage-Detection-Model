"""
ROI extraction and preprocessing utilities.
Handles cropping, resizing, and normalization.
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple
from config import img_size, norm_mean, norm_std

log = logging.getLogger(__name__)


class ROI:
    """Extract and preprocess regions of interest."""
    
    def __init__(self, size: int = img_size):
        """
        Initialize ROI processor.
        
        Args:
            size: Target image size (square)
        """
        self.size = size
    
    def crop(self, path: str, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop region of interest from image.
        
        Args:
            path: Path to image file
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped image (original size)
        """
        try:
            img = cv2.imread(str(path))
            if img is None:
                log.error(f"Failed to load: {path}")
                return None
            
            x1, y1, x2, y2 = box
            h, w = img.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            roi = img[y1:y2, x1:x2]
            return roi
        except Exception as e:
            log.error(f"Error cropping: {str(e)}")
            return None
    
    def resize(self, roi: np.ndarray) -> np.ndarray:
        """
        Resize ROI to target size.
        
        Args:
            roi: Image region
            
        Returns:
            Resized image
        """
        if roi is None or roi.size == 0:
            return None
        
        resized = cv2.resize(roi, (self.size, self.size), 
                            interpolation=cv2.INTER_LINEAR)
        return resized
    
    def normalize(self, roi: np.ndarray, 
                 mean: list = None, std: list = None) -> np.ndarray:
        """
        Normalize ROI using ImageNet statistics.
        
        Args:
            roi: Image region (BGR from OpenCV)
            mean: Normalization mean (per channel)
            std: Normalization std (per channel)
            
        Returns:
            Normalized image
        """
        if roi is None:
            return None
        
        if mean is None or std is None:
            mean = norm_mean
            std = norm_std
        
        # BGR to RGB and scale to [0, 1]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Normalize per channel
        roi[:, :, 0] = (roi[:, :, 0] - mean[0]) / std[0]
        roi[:, :, 1] = (roi[:, :, 1] - mean[1]) / std[1]
        roi[:, :, 2] = (roi[:, :, 2] - mean[2]) / std[2]
        
        return roi
    
    def process(self, path: str, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Complete pipeline: crop, resize, normalize.
        
        Args:
            path: Path to image file
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Preprocessed ROI
        """
        roi = self.crop(path, box)
        if roi is not None:
            roi = self.resize(roi)
            roi = self.normalize(roi)
        return roi
