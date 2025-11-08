import json
import numpy as np
import cv2
from os.path import isfile, join
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config.constants import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED

class COCODamageDataset(Dataset):
    """
    Complete rewrite to support COCO JSON annotations instead of masks.
    Loads COCO format data with per-instance annotations.
    """
    
    def __init__(self, image_dir: str, annotations_file: str, image_preprocessor=None, data_augmentor=None):
        self.image_dir = Path(image_dir)
        self.image_preprocessor = image_preprocessor
        self.data_augmentor = data_augmentor
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create ID mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations_by_image = {}
        
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
        
        # Class mapping from COCO categories
        self.category_id_to_class = {cat['id']: idx for idx, cat in enumerate(self.coco_data['categories'])}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = self.image_dir / image_info['file_name']
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        height, width = image.shape[:2]
        
        bboxes = []
        class_ids = []
        
        if image_id in self.annotations_by_image:
            for ann in self.annotations_by_image[image_id]:
                bbox = ann['bbox']  # [x, y, width, height]
                
                # Convert to normalized coordinates [x_center, y_center, width, height]
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                norm_width = bbox[2] / width
                norm_height = bbox[3] / height
                
                class_id = self.category_id_to_class.get(ann['category_id'], 0)
                bboxes.append([x_center, y_center, norm_width, norm_height])
                class_ids.append(class_id)
        
        # Preprocess
        if self.image_preprocessor:
            image = self.image_preprocessor.process(image)
        
        # Augment
        if self.data_augmentor and bboxes:
            image, bboxes = self.data_augmentor.apply_augmentation(image, bboxes)
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4), dtype=torch.float32)
        class_ids_tensor = torch.tensor(class_ids, dtype=torch.long) if class_ids else torch.zeros((0,), dtype=torch.long)
        
        return {
            'image': image_tensor,
            'bboxes': bboxes_tensor,
            'class_ids': class_ids_tensor,
            'num_bboxes': len(bboxes),
            'image_id': image_id,
            'image_path': str(image_path)
        }


class DatasetSplitter:
    @staticmethod
    def divide_into_splits(annotation_file: str, base_image_dir: str,
                          train_split: float = TRAIN_SPLIT,
                          val_split: float = VAL_SPLIT) -> Dict[str, Tuple[str, str]]:
        """
        Reorganized to use COCO annotation files directly.
        Returns paths to train/val/test annotation files and image directories.
        """
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Get all image IDs
        image_ids = [img['id'] for img in coco_data['images']]
        
        # Split data 70-15-15
        train_ids, temp_ids = train_test_split(
            image_ids,
            test_size=(1 - train_split),
            random_state=RANDOM_SEED
        )
        
        val_ratio = val_split / (1 - train_split)
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1 - val_ratio),
            random_state=RANDOM_SEED
        )
        
        # Create split annotations
        splits = {
            'train': DatasetSplitter._create_split_annotation(coco_data, set(train_ids)),
            'val': DatasetSplitter._create_split_annotation(coco_data, set(val_ids)),
            'test': DatasetSplitter._create_split_annotation(coco_data, set(test_ids))
        }
        
        return splits, base_image_dir

    @staticmethod
    def _create_split_annotation(coco_data: Dict, image_ids_set: set) -> Dict:
        """Create a subset of COCO annotations for given image IDs."""
        split_images = [img for img in coco_data['images'] if img['id'] in image_ids_set]
        split_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids_set]
        
        return {
            'images': split_images,
            'annotations': split_annotations,
            'categories': coco_data['categories']
        }

    @staticmethod
    def build_dataloaders(train_annotation: Dict, val_annotation: Dict, test_annotation: Dict,
                         base_image_dir: str,
                         batch_size_value: int = 16,
                         image_preprocessor=None,
                         data_augmentor=None) -> Dict[str, DataLoader]:
        """
        Updated to use new COCODamageDataset with annotation dicts.
        """
        
        import tempfile
        import json
        
        temp_dir = Path(tempfile.gettempdir()) / 'coco_splits'
        temp_dir.mkdir(exist_ok=True)
        
        train_ann_file = temp_dir / 'train_annotations.json'
        val_ann_file = temp_dir / 'val_annotations.json'
        test_ann_file = temp_dir / 'test_annotations.json'
        
        with open(train_ann_file, 'w') as f:
            json.dump(train_annotation, f)
        with open(val_ann_file, 'w') as f:
            json.dump(val_annotation, f)
        with open(test_ann_file, 'w') as f:
            json.dump(test_annotation, f)
        
        image_dir = Path(base_image_dir)
        
        training_dataset = COCODamageDataset(
            image_dir, str(train_ann_file),
            image_preprocessor=image_preprocessor,
            data_augmentor=data_augmentor
        )
        
        validation_dataset = COCODamageDataset(
            image_dir, str(val_ann_file),
            image_preprocessor=image_preprocessor,
            data_augmentor=None
        )
        
        testing_dataset = COCODamageDataset(
            image_dir, str(test_ann_file),
            image_preprocessor=image_preprocessor,
            data_augmentor=None
        )
        
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=batch_size_value,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn_coco
        )
        
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size_value,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn_coco
        )
        
        testing_dataloader = DataLoader(
            testing_dataset,
            batch_size=batch_size_value,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn_coco
        )
        
        return {
            'train': training_dataloader,
            'val': validation_dataloader,
            'test': testing_dataloader
        }


def collate_fn_coco(batch):
    """
    Custom collate function to handle variable number of bboxes per image.
    """
    images = torch.stack([item['image'] for item in batch])
    
    return {
        'images': images,
        'bboxes': [item['bboxes'] for item in batch],
        'class_ids': [item['class_ids'] for item in batch],
        'num_bboxes': [item['num_bboxes'] for item in batch],
        'image_ids': [item['image_id'] for item in batch],
        'image_paths': [item['image_path'] for item in batch]
    }


def load_single_image_with_bbox(image_path: str, bbox_path: str = None) -> tuple:
    """
    Load a single image and optional bounding boxes without batch processing.
    """
    image_array = cv2.imread(image_path)
    if image_array is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    bboxes = []
    if bbox_path and isfile(bbox_path):
        with open(bbox_path, 'r') as bbox_file:
            bboxes = [list(map(float, line.strip().split())) for line in bbox_file if line.strip()]
    
    return image_array, bboxes
