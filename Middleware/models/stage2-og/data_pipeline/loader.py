import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from config.constants import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED

class Stage2DamageDataset(Dataset):
    def __init__(self, image_list: list, bbox_list: list, image_preprocessor=None, data_augmentor=None):
        self.image_list = image_list
        self.bbox_list = bbox_list
        self.image_preprocessor = image_preprocessor
        self.data_augmentor = data_augmentor

        assert len(image_list) == len(bbox_list), "Image and bbox counts must match"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, dataset_index):
        current_image = self.image_list[dataset_index]
        current_bboxes = self.bbox_list[dataset_index]

        if self.image_preprocessor:
            current_image = self.image_preprocessor.process(current_image)

        if self.data_augmentor:
            current_image, current_bboxes = self.data_augmentor.apply_augmentation(current_image, current_bboxes)

        image_tensor = torch.from_numpy(current_image).permute(2, 0, 1)
        bboxes_tensor = torch.tensor(current_bboxes, dtype=torch.float32)

        return {
            'image': image_tensor,
            'bboxes': bboxes_tensor,
            'num_bboxes': len(current_bboxes)
        }


class DatasetSplitter:
    @staticmethod
    def divide_into_splits(image_collection: list, bbox_collection: list, 
                          training_fraction: float = TRAIN_SPLIT,
                          validation_fraction: float = VAL_SPLIT) -> dict:
        training_images, temporary_images, training_bboxes, temporary_bboxes = train_test_split(
            image_collection, bbox_collection,
            test_size=(1 - training_fraction),
            random_state=RANDOM_SEED
        )

        validation_ratio = validation_fraction / (1 - training_fraction)
        validation_images, testing_images, validation_bboxes, testing_bboxes = train_test_split(
            temporary_images, temporary_bboxes,
            test_size=(1 - validation_ratio),
            random_state=RANDOM_SEED
        )

        return {
            'train': (training_images, training_bboxes),
            'val': (validation_images, validation_bboxes),
            'test': (testing_images, testing_bboxes)
        }

    @staticmethod
    def build_dataloaders(training_data_split: tuple, validation_data_split: tuple, 
                         testing_data_split: tuple,
                         batch_size_value: int = 16,
                         image_preprocessor=None, 
                         data_augmentor=None) -> dict:
        training_dataset = Stage2DamageDataset(
            training_data_split[0], training_data_split[1],
            image_preprocessor=image_preprocessor,
            data_augmentor=data_augmentor
        )

        validation_dataset = Stage2DamageDataset(
            validation_data_split[0], validation_data_split[1],
            image_preprocessor=image_preprocessor,
            data_augmentor=None
        )

        testing_dataset = Stage2DamageDataset(
            testing_data_split[0], testing_data_split[1],
            image_preprocessor=image_preprocessor,
            data_augmentor=None
        )

        training_dataloader = DataLoader(
            training_dataset,
            batch_size=batch_size_value,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size_value,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        testing_dataloader = DataLoader(
            testing_dataset,
            batch_size=batch_size_value,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return {
            'train': training_dataloader,
            'val': validation_dataloader,
            'test': testing_dataloader
        }
