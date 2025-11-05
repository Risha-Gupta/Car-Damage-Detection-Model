import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from config.constants import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED

class Stage2Dataset(Dataset):
    def __init__(self, images: list, bboxes_list: list, preprocessor=None, transform=None):
        self.images = images
        self.bboxes_list = bboxes_list
        self.preprocessor = preprocessor
        self.transform = transform

        assert len(images) == len(bboxes_list), "Image and bbox counts must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        bboxes = self.bboxes_list[index]

        if self.preprocessor:
            image = self.preprocessor.process(image)

        if self.transform:
            image, bboxes = self.transform(image, bboxes)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)

        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

        return {
            'image': image_tensor,
            'bboxes': bboxes_tensor,
            'num_bboxes': len(bboxes)
        }


class DataSplitter:
    @staticmethod
    def split_data(images: list, bboxes_list: list, 
                   train_split: float = TRAIN_SPLIT,
                   val_split: float = VAL_SPLIT) -> dict:
        train_images, temp_images, train_bboxes, temp_bboxes = train_test_split(
            images, bboxes_list,
            test_size=(1 - train_split),
            random_state=RANDOM_SEED
        )

        val_ratio = val_split / (1 - train_split)
        val_images, test_images, val_bboxes, test_bboxes = train_test_split(
            temp_images, temp_bboxes,
            test_size=(1 - val_ratio),
            random_state=RANDOM_SEED
        )

        return {
            'train': (train_images, train_bboxes),
            'val': (val_images, val_bboxes),
            'test': (test_images, test_bboxes)
        }

    @staticmethod
    def create_dataloaders(train_data: tuple, val_data: tuple, test_data: tuple,
                          batch_size: int = 16,
                          preprocessor=None, augmentor=None) -> dict:
        train_dataset = Stage2Dataset(
            train_data[0], train_data[1],
            preprocessor=preprocessor,
            transform=augmentor
        )

        val_dataset = Stage2Dataset(
            val_data[0], val_data[1],
            preprocessor=preprocessor,
            transform=None
        )

        test_dataset = Stage2Dataset(
            test_data[0], test_data[1],
            preprocessor=preprocessor,
            transform=None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
