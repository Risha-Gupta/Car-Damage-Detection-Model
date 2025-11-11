import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import albumentations as A
from pathlib import Path
import json
import sys

from config import *
from data_loader import DataManager
from augmentation import get_train_augmentation, get_val_augmentation
from model import ComponentClassifier, FocalLoss
from trainer import Trainer
from output_handler import LogHandler


def main():
    """Main training script."""
    print("=" * 60)
    print("Stage 4: Component Classification Training")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data loading
    print("\n1. Loading data...")
    data_mgr = DataManager(data_root, train_split, val_split, test_split)
    
    # You would load your COCO annotations here
    # For this example, we'll use placeholder logic
    anno_file = Path(data_root) / 'annotations.json'
    if not anno_file.exists():
        print(f"Error: Annotation file not found at {anno_file}")
        print("Please ensure your COCO dataset is properly set up.")
        sys.exit(1)
    
    images, labels, damages = data_mgr.load_coco_data(anno_file)
    print(f"Total images: {len(images)}")
    print(f"Component classes: {len(data_mgr.comp_names)}")
    
    # Balance classes
    print("\n2. Balancing classes...")
    images, labels, damages = data_mgr.balance_classes(images, labels, damages)
    print(f"After balancing: {len(images)} images")
    
    # Split data
    print("\n3. Splitting data...")
    train_data, val_data, test_data = data_mgr.split_data(
        images, labels, damages
    )
    
    print(f"Train: {len(train_data['images'])}")
    print(f"Val: {len(val_data['images'])}")
    print(f"Test: {len(test_data['images'])}")
    
    # Augmentation
    print("\n4. Setting up augmentation...")
    train_aug = get_train_augmentation(
        img_size, rotation=aug_rotation, zoom=aug_zoom,
        brightness=aug_brightness, contrast=aug_contrast,
        hflip=aug_hflip, vflip=aug_vflip
    )
    val_aug = get_val_augmentation(img_size)
    
    # Create datasets (placeholder - you'd load actual images)
    # For now, we'll create dummy tensors
    train_x = torch.randn(len(train_data['images']), 3, img_size, img_size)
    train_y = torch.tensor(train_data['labels'], dtype=torch.long)
    
    val_x = torch.randn(len(val_data['images']), 3, img_size, img_size)
    val_y = torch.tensor(val_data['labels'], dtype=torch.long)
    
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    print("\n5. Creating model...")
    num_classes = len(data_mgr.comp_names)
    model = ComponentClassifier(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained
    )
    
    if freeze_backbone:
        model.freeze_backbone()
        print("Backbone frozen")
    
    # Trainer
    print("\n6. Starting training...")
    trainer = Trainer(model, device=device, use_focal=use_focal_loss,
                     alpha=focal_alpha, gamma=focal_gamma)
    trainer.setup_optimizer(lr=learning_rate, optimizer=optimizer_name,
                           weight_decay=weight_decay)
    
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=num_epochs,
        early_stop_patience=early_stop_patience,
        model_dir=model_root
    )
    
    # Save logs
    print("\n7. Saving results...")
    handler = LogHandler(output_root)
    history_df = trainer.get_history_df()
    handler.save_training_log(history_df, model_name='component_classifier')
    handler.save_metrics_summary(history_df, {}, model_name='component_classifier')
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
