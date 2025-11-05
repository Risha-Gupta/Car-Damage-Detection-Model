import numpy as np
from albumentations import (
    Compose, HorizontalFlip, Rotate, 
    RandomBrightnessContrast, GaussNoise,
    Resize, BboxParams
)
import cv2
from config.constants import AUGMENTATION_CONFIG

class DataAugmentor:
    def __init__(self, config: dict = None):
        if config is None:
            config = AUGMENTATION_CONFIG

        self.augmentor = Compose([
            HorizontalFlip(p=config.get('horizontal_flip', 0.5)),
            Rotate(limit=config.get('rotation_limit', 15), p=0.5),
            RandomBrightnessContrast(
                brightness_limit=config.get('brightness_limit', 0.3),
                contrast_limit=0.2,
                p=0.5
            ),
            GaussNoise(p=config.get('gaussian_noise', 0.3)),
            Resize(
                height=config.get('target_size', (512, 512))[0],
                width=config.get('target_size', (512, 512))[1],
                always_apply=True
            )
        ],
        bbox_params=BboxParams(format='pascal_voc', min_area=0.0, min_visibility=0.0)
        )

    def augment(self, image: np.ndarray, bboxes: list) -> tuple:
        try:
            transformed = self.augmentor(
                image=image,
                bboxes=bboxes
            )

            augmented_image = transformed['image']
            augmented_bboxes = transformed['bboxes']

            return augmented_image, augmented_bboxes
        except Exception as e:
            return image, bboxes

    def create_augmented_dataset(self, images: list, bboxes_list: list, 
                                multiplier: int = 3) -> tuple:
        augmented_images = list(images)
        augmented_bboxes = list(bboxes_list)

        for i in range(multiplier):
            for image, bboxes in zip(images, bboxes_list):
                aug_image, aug_bboxes = self.augment(image, bboxes)
                augmented_images.append(aug_image)
                augmented_bboxes.append(aug_bboxes)

        return augmented_images, augmented_bboxes
