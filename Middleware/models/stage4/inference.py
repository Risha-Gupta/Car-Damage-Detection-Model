import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from config import *
from model import ComponentClassifier, EnsembleClassifier
from augmentation import get_val_augmentation
from output_handler import LogHandler


def predict_single(model, img_path, comp_names, aug_transform, device):
    """Predict on single image."""
    start_time = time.time()
    
    # Load image
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None
    
    img_array = np.array(img)
    
    # Augment
    if aug_transform:
        transformed = aug_transform(image=img_array)
        img_tensor = transformed['image'].unsqueeze(0)
    else:
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
    
    img_tensor = img_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
    
    pred_idx = probs.argmax(dim=1).item()
    pred_score = probs[0, pred_idx].item()
    
    elapsed = time.time() - start_time
    
    return {
        'image': Path(img_path).name,
        'label': comp_names[pred_idx],
        'score': round(pred_score, 4),
        'time': round(elapsed, 4)
    }


def inference_on_folder(model_path, img_folder, comp_names, device='cpu'):
    """Run inference on all images in folder."""
    print("=" * 60)
    print("Stage 4: Component Classification Inference")
    print("=" * 60)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    num_classes = len(comp_names)
    model = ComponentClassifier(backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get augmentation
    aug = get_val_augmentation(img_size)
    
    # Get images
    img_folder = Path(img_folder)
    img_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    img_paths = []
    
    for fmt in img_formats:
        img_paths.extend(img_folder.glob(f"*{fmt}"))
        img_paths.extend(img_folder.glob(f"*{fmt.upper()}"))
    
    print(f"Found {len(img_paths)} images")
    
    if not img_paths:
        print("No images found!")
        return
    
    # Inference
    print("\nRunning inference...")
    results = []
    
    for img_path in tqdm(img_paths):
        result = predict_single(model, img_path, comp_names, aug, device)
        if result:
            results.append(result)
    
    # Save results
    print("\nSaving results...")
    handler = LogHandler(output_root)
    handler.save_inference_log(results, model_name='component_classifier')
    
    print("\n" + "=" * 60)
    print(f"Inference complete! Processed {len(results)} images")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <model_path> [image_folder]")
        print("Example: python inference.py ./models/best_model.pt ./images")
        sys.exit(1)
    
    model_path = sys.argv[1]
    img_folder = sys.argv[2] if len(sys.argv) > 2 else './images'
    
    # Load component names from your dataset
    comp_names = [f"component_{i}" for i in range(10)]  # Placeholder
    
    inference_on_folder(model_path, img_folder, comp_names)
