"""
Data augmentation utilities using Albumentations.
Provides augmentation pipelines for training and validation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_aug(img_size, rotation=20, zoom=0.2, brightness=0.2, 
                  contrast=0.2, hflip=True, vflip=False):
    """
    Get augmentation pipeline for training.
    Includes random transforms for robustness.
    """
    return A.Compose([
        A.Rotate(limit=rotation, p=0.5),
        A.Perspective(scale=(zoom, zoom), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=brightness,
            contrast_limit=contrast,
            p=0.5
        ),
        A.GaussNoise(p=0.2),
        A.HorizontalFlip(p=0.5) if hflip else A.NoOp(),
        A.VerticalFlip(p=0.3) if vflip else A.NoOp(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        A.Resize(img_size, img_size),
        ToTensorV2()
    ])


def get_val_aug(img_size):
    """
    Get augmentation pipeline for validation.
    Only normalization and resizing, no random transforms.
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        A.Resize(img_size, img_size),
        ToTensorV2()
    ])


def get_test_aug(img_size):
    """Get augmentation pipeline for testing (same as validation)."""
    return get_val_aug(img_size)
