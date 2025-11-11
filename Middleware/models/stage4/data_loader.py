"""
Data loading utilities for Stage 4 pipeline.
Handles COCO annotations and Stage 3 detection results.
"""

import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from typing import Dict, List, Tuple, Any
import logging

log = logging.getLogger(__name__)


class CocoLoader:
    """Load COCO-style annotations from JSON."""
    
    def __init__(self, ann_file: str):
        """
        Initialize COCO loader.
        
        Args:
            ann_file: Path to COCO JSON annotation file
        """
        self.ann_file = ann_file
        self.data = self._load()
        self.imgs = {img['id']: img for img in self.data.get('images', [])}
        self.cats = {cat['id']: cat for cat in self.data.get('categories', [])}
    
    def _load(self) -> Dict[str, Any]:
        """Load and parse COCO JSON file."""
        try:
            with open(self.ann_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            log.error(f"File not found: {self.ann_file}")
            raise
        except json.JSONDecodeError:
            log.error(f"Invalid JSON: {self.ann_file}")
            raise
    
    def get_image(self, img_id: int) -> Dict[str, Any]:
        """Get image info by ID."""
        return self.imgs.get(img_id, {})
    
    def get_category(self, cat_id: int) -> str:
        """Get category name by ID."""
        return self.cats.get(cat_id, {}).get('name', 'unknown')
    
    def get_annotations(self, img_id: int) -> List[Dict[str, Any]]:
        """Get all annotations for an image."""
        return [ann for ann in self.data.get('annotations', []) 
                if ann['image_id'] == img_id]


class Stage3Loader:
    """Load Stage 3 bounding box detection results."""
    
    def __init__(self, results_file: str):
        """
        Initialize Stage 3 results loader.
        
        Args:
            results_file: Path to Stage 3 results JSON file
        """
        self.file = results_file
        self.all_detections = self._load()
        self._organize()
    
    def _load(self) -> List[Dict[str, Any]]:
        """Load Stage 3 results."""
        try:
            with open(self.file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            log.warning(f"Results file not found: {self.file}")
            return []
        except json.JSONDecodeError:
            log.error(f"Invalid JSON: {self.file}")
            raise
    
    def _organize(self):
        """Organize results by image name."""
        self.by_image = {}
        for det in self.all_detections:
            name = det.get('image_name', '')
            if name not in self.by_image:
                self.by_image[name] = []
            self.by_image[name].append(det)
    
    def get(self, img_name: str) -> List[Dict[str, Any]]:
        """Get detections for an image."""
        return self.by_image.get(img_name, [])
    
    def box(self, det: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Extract bbox from detection.
        Returns [x1, y1, x2, y2] format.
        """
        bbox = det.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            return (int(x), int(y), int(x + w), int(y + h))
        return tuple(map(int, bbox))
    
    def damage_type(self, det: Dict[str, Any]) -> str:
        """Get damage type from detection."""
        return det.get('damage_type', 'unknown')
    
    def score(self, det: Dict[str, Any]) -> float:
        """Get confidence score from detection."""
        return float(det.get('confidence', 0.0))


class DataManager:
    """
    Manage data loading with class balancing and stratified splitting.
    """
    
    def __init__(self, data_root, train_split=0.70, val_split=0.15, test_split=0.15):
        """Initialize data manager."""
        self.data_root = data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
    
    def balance_classes(self, images, labels):
        """
        Apply class balancing by oversampling minority classes.
        Ensures all classes have equal representation.
        """
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        
        balanced_imgs = list(images)
        balanced_lbls = list(labels)
        
        for label, count in label_counts.items():
            if count < max_count:
                # Find indices of this class
                idx = [i for i, l in enumerate(labels) if l == label]
                need = max_count - count
                
                # Oversample
                oversample_idx = np.random.choice(idx, size=need, replace=True)
                for i in oversample_idx:
                    balanced_imgs.append(images[i])
                    balanced_lbls.append(labels[i])
        
        return balanced_imgs, balanced_lbls
    
    def split_data(self, images, labels, seed=42):
        """
        Split data into train/val/test with stratification.
        Maintains class distribution across splits.
        """
        np.random.seed(seed)
        n = len(images)
        
        # Use StratifiedKFold for balanced splits
        skf = StratifiedKFold(n_splits=int(1/self.test_split), 
                             shuffle=True, random_state=seed)
        
        train_idx, test_idx = next(skf.split(np.arange(n), labels))
        
        # Split train into train/val
        skf2 = StratifiedKFold(n_splits=int(1/self.val_split), 
                              shuffle=True, random_state=seed)
        train_idx_sub, val_idx_sub = next(skf2.split(
            train_idx, 
            [labels[i] for i in train_idx]
        ))
        
        val_idx = train_idx[val_idx_sub]
        train_idx = train_idx[train_idx_sub]
        
        return {
            'train': {'images': [images[i] for i in train_idx], 
                     'labels': [labels[i] for i in train_idx]},
            'val': {'images': [images[i] for i in val_idx], 
                   'labels': [labels[i] for i in val_idx]},
            'test': {'images': [images[i] for i in test_idx], 
                    'labels': [labels[i] for i in test_idx]}
        }
    
    def create_kfolds(self, images, labels, n_folds=5, seed=42):
        """
        Create K-fold splits for cross-validation.
        Returns list of folds, each with train/val split.
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        
        for train_idx, val_idx in skf.split(images, labels):
            fold = {
                'train': {'images': [images[i] for i in train_idx], 
                         'labels': [labels[i] for i in train_idx]},
                'val': {'images': [images[i] for i in val_idx], 
                       'labels': [labels[i] for i in val_idx]}
            }
            folds.append(fold)
        
        return folds
