"""
Configuration file for Stage 4 car component damage classification pipeline.
Centralized settings for paths, model parameters, and processing options.
"""

import os
from pathlib import Path

# Dataset paths
data_root = "./CarDD_release/CarDD_COCO"
train_imgs = os.path.join(data_root, "train2017")
val_imgs = os.path.join(data_root, "val2017")
test_imgs = os.path.join(data_root, "test2017")

ann_dir = os.path.join(data_root, "annotations")
train_ann = os.path.join(ann_dir, "instances_train2017.json")
val_ann = os.path.join(ann_dir, "instances_val2017.json")
test_ann = os.path.join(ann_dir, "instances_test2017.json")

# Stage 3 detection results
stage3_file = "./stage3_detections.json"

# Model paths
model_path = "./models/component_classifier.pt"
model_type = "vit"  # "vit", "efficientnet", or "resnet"
img_size = 224
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# Training configuration
train_split = 0.70
val_split = 0.15
test_split = 0.15
batch_size = 32
workers = 4
num_epochs = 50
learning_rate = 1e-4
weight_decay = 0.001
dropout_rate = 0.4
early_stop_patience = 5
k_folds = 5

# Data augmentation settings
aug_rotation = 20
aug_zoom = 0.2
aug_brightness = 0.2
aug_contrast = 0.2
aug_hflip = True
aug_vflip = False

# Loss and regularization
use_focal_loss = True
focal_alpha = 0.25
focal_gamma = 2.0
use_dropout = True
use_batch_norm = True

# Ensemble configuration
use_ensemble = True
num_models = 3
model_weights = None  # None for equal weights, or list like [0.5, 0.3, 0.2]

# Component labels for classification
components = [
    "front_bumper",
    "rear_bumper",
    "door_front_left",
    "door_front_right",
    "door_rear_left",
    "door_rear_right",
    "hood",
    "trunk",
    "fender_left",
    "fender_right",
    "windshield",
    "side_window",
    "headlight_left",
    "headlight_right",
    "taillight_left",
    "taillight_right",
    "side_mirror_left",
    "side_mirror_right",
    "wheel_left",
    "wheel_right",
    "engine",
    "undercarriage"
]

# Damage type labels
damages = [
    "dent",
    "crack",
    "scratch",
    "broken",
    "missing"
]

# Processing settings
device = "cuda"  # "cuda" or "cpu"
conf_threshold = 0.5

# Output
out_dir = "./output"
excel_file = os.path.join(out_dir, "results.xlsx")
log_dir = os.path.join(out_dir, "logs")
model_dir = os.path.join(out_dir, "models")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
