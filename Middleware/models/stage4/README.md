# Stage 4: Car Component Damage Classification

Complete pipeline for training and deploying a component damage classifier with best practices for accuracy and reproducibility.

## New Features (Enhanced Version)

- **Augmentation**: Rotation, zoom, brightness/contrast, and horizontal flips using Albumentations
- **Class Balancing**: Automatic oversampling of underrepresented damage categories
- **Focal Loss**: Specialized loss for imbalanced classification
- **Transfer Learning**: Pretrained EfficientNet, ResNet, or ViT backbones
- **Regularization**: Dropout (0.4), BatchNorm, Weight Decay, Early Stopping
- **K-Fold Cross-Validation**: For robust model evaluation
- **Comprehensive Logging**: Excel output with training metrics and inference results (4 decimal precision)

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Configuration (config.py)

Key parameters:

- **Training**: batch_size=32, learning_rate=1e-4, num_epochs=50
- **Data Split**: 70% train, 15% val, 15% test (stratified)
- **Augmentation**: rotation=20°, zoom=0.2, brightness/contrast=0.2
- **Regularization**: dropout=0.4, weight_decay=0.001
- **Loss**: Focal loss (alpha=0.25, gamma=2.0) for imbalanced data
- **Model**: EfficientNetB3 with ImageNet pretraining

## Training

\`\`\`bash
python train_component_classifier.py
\`\`\`

Outputs:
- Best model: \`./models/best_model.pt\`
- Training log: \`./output/training_log_*.xlsx\` with columns:
  - Epoch, Train_Loss, Val_Loss, Train_Acc, Val_Acc, GPU_Mem (GB)

Example training output:

| Epoch | Train_Loss | Val_Loss | Train_Acc | Val_Acc | GPU_Mem |
|-------|-----------|----------|-----------|---------|---------|
| 1     | 0.8234    | 0.7891   | 0.6234    | 0.6541  | 2.1345  |
| 2     | 0.7123    | 0.6945   | 0.7123    | 0.7234  | 2.1345  |
| 5     | 0.5432    | 0.5123   | 0.8234    | 0.8145  | 2.1345  |

## Inference

\`\`\`bash
python run_inference.py --split test
\`\`\`

Outputs inference log Excel file with columns:
- Image_Name, Component_Label, Component_Score, Execution_Time

Example inference results:

| Image_Name | Component_Label | Component_Score | Execution_Time |
|-----------|-----------------|-----------------|-----------------|
| car_001.jpg | front_bumper   | 0.9345          | 0.0234          |
| car_002.jpg | door_front_left| 0.8756          | 0.0245          |

## Key Best Practices Implemented

### Data Level
✓ Albumentations pipeline (rotation, zoom, brightness/contrast)
✓ Automatic class balancing via oversampling
✓ ImageNet normalization ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
✓ Stratified 70/15/15 train/val/test split

### Model Level
✓ Transfer learning (EfficientNetB3 default)
✓ Pretrained ImageNet weights
✓ Focal Loss for imbalanced data
✓ Learnable head with BatchNorm and Dropout

### Training Level
✓ Adam optimizer (lr=1e-4, weight_decay=0.001)
✓ StepLR scheduler (decay 0.1 every 10 epochs)
✓ Early stopping (patience=5)
✓ Dropout 0.4 and BatchNorm regularization

### Validation Level
✓ Separate train/val/test splits
✓ Stratified splitting to maintain class balance
✓ Metrics logged to 4 decimal precision
✓ GPU memory tracking

## Output Files

**Training Log** (training_log_component_classifier_*.xlsx):
- Epoch-by-epoch loss and accuracy metrics
- GPU memory usage tracking
- Used to identify overfitting/underfitting

**Inference Log** (inference_log_component_classifier_*.xlsx):
- Per-image predictions with confidence scores
- Execution time per inference
- Used for deployment performance analysis

**Stage 4 Results** (stage4_results_stage4_*.xlsx):
- ID, Image Name, Timestamp, ROI coordinates
- Damage Type, Component Label, Confidence Score
- Execution Time in seconds

## Customization

### Change Backbone

Edit config.py:
\`\`\`python
model_type = 'resnet50'  # or 'vit_base'
\`\`\`

### Adjust Augmentation

Edit config.py:
\`\`\`python
aug_rotation = 30        # More rotation
aug_brightness = 0.3     # More brightness variation
\`\`\`

### Tune Hyperparameters

Edit config.py:
\`\`\`python
batch_size = 16          # Smaller for low VRAM
learning_rate = 5e-5     # Lower for fine-tuning
dropout_rate = 0.3       # Less dropout
\`\`\`

## Troubleshooting

**CUDA out of memory**: Reduce batch_size or use smaller backbone (ResNet50)
**Low accuracy**: Increase num_epochs, reduce learning_rate, add more augmentation
**Overfitting**: Increase dropout_rate or weight_decay
**Slow training**: Use smaller img_size (e.g., 192x192) or enable mixed precision

## Performance Tips

- GPU: Model auto-uses CUDA if available
- 8GB VRAM: Set batch_size=16
- Limited VRAM: Use ResNet50 instead of EfficientNetB3
- Faster inference: Use lower img_size (with accuracy trade-off)

## References

- EfficientNet: https://arxiv.org/abs/1905.11946
- Focal Loss: https://arxiv.org/abs/1708.02002
- Albumentations: https://github.com/albumentations-team/albumentations

---

For questions, check config.py for parameter definitions or review docstrings in each module.
\`\`\`
