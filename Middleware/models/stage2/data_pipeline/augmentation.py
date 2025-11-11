import numpy as np
from albumentations import (
    Compose, HorizontalFlip, Rotate, 
    RandomBrightnessContrast, GaussNoise,
    Resize, BboxParams
)
from config.constants import AUGMENTATION_CONFIG

class DataAugmentor:
    def __init__(self, augmentation_settings: dict = None):
        if augmentation_settings is None:
            augmentation_settings = AUGMENTATION_CONFIG

        self.transformation_pipeline = Compose([
            HorizontalFlip(p=augmentation_settings.get('horizontal_flip', 0.5)),
            Rotate(limit=augmentation_settings.get('rotation_limit', 15), p=0.5),
            RandomBrightnessContrast(
                brightness_limit=augmentation_settings.get('brightness_limit', 0.3),
                contrast_limit=0.2,
                p=0.5
            ),
            GaussNoise(p=augmentation_settings.get('gaussian_noise', 0.3)),
            Resize(
                height=augmentation_settings.get('target_size', (512, 512))[0],
                width=augmentation_settings.get('target_size', (512, 512))[1],
                always_apply=True
            )
        ],
        bbox_params=BboxParams(format='pascal_voc', min_area=0.0, min_visibility=0.0)
        )

    def apply_augmentation(self, image: np.ndarray, bounding_boxes: list) -> tuple:
        try:
            transformation_result = self.transformation_pipeline(
                image=image,
                bboxes=bounding_boxes
            )
            augmented_image = transformation_result['image']
            augmented_bounding_boxes = transformation_result['bboxes']
            return augmented_image, augmented_bounding_boxes
        except Exception as error_occurred:
            return image, bounding_boxes

    def expand_dataset_with_augmentation(self, image_collection: list, 
                                        bbox_collection: list, 
                                        multiplication_factor: int = 3) -> tuple:
        expanded_images = list(image_collection)
        expanded_bboxes = list(bbox_collection)

        for iteration_count in range(multiplication_factor):
            for image_data, bbox_data in zip(image_collection, bbox_collection):
                transformed_image, transformed_bboxes = self.apply_augmentation(image_data, bbox_data)
                expanded_images.append(transformed_image)
                expanded_bboxes.append(transformed_bboxes)

        return expanded_images, expanded_bboxes
