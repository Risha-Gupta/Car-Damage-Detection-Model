import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import time
from pathlib import Path
from typing import Dict
import numpy as np
import cv2
from os import listdir
from os.path import join, isfile

from models.yolov8_detector import DamageLocalizationModel
from data_pipeline.preprocessor import ImagePreprocessor
from data_pipeline.augmentation import DataAugmentor
from data_pipeline.loader import DatasetSplitter, Stage2DamageDataset
from logging.performance_logger import PerformanceLogger
from logging.excel_reporter import ExcelReporter
from config.constants import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_MAX_NORM,
    DATA_DIR, CHECKPOINT_DIR, LOG_DIR, REPORT_OUTPUT,
    AUGMENTATION_CONFIG, INPUT_SIZE
)


class DamageLocalizationTrainer:
    def __init__(self, detector_model: DamageLocalizationModel, training_config: Dict,
                 training_dataloader, validation_dataloader):
        self.detector_model = detector_model
        self.training_configuration = training_config
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader

        self.bbox_loss_function = nn.SmoothL1Loss()

        self.optimizer_instance = Adam(
            self.detector_model.get_model_instance().parameters(),
            lr=training_config.get('learning_rate', LEARNING_RATE),
            weight_decay=training_config.get('weight_decay', WEIGHT_DECAY)
        )

        self.scheduler_instance = ReduceLROnPlateau(
            self.optimizer_instance,
            mode='min',
            factor=training_config.get('scheduler_factor', 0.1),
            patience=training_config.get('scheduler_patience', 3),
            min_lr=training_config.get('min_lr', 1e-5)
        )

        self.best_validation_loss_value = float('inf')
        self.patience_threshold = training_config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
        self.patience_counter = 0

        self.performance_tracker = PerformanceLogger()
        self.result_exporter = ExcelReporter()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def train_single_epoch(self, epoch_number: int) -> float:
        self.detector_model.get_model_instance().train()
        cumulative_epoch_loss = 0

        for batch_index, batch_content in enumerate(self.training_dataloader):
            image_tensor = batch_content['image'].to(self.device)
            bbox_tensor = batch_content['bboxes'].to(self.device)

            batch_timer_start = time.time()

            predicted_bboxes = self.detector_model.get_model_instance()(image_tensor)
            computed_loss = self.bbox_loss_function(predicted_bboxes, bbox_tensor)

            self.optimizer_instance.zero_grad()
            computed_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.detector_model.get_model_instance().parameters(),
                max_norm=GRADIENT_CLIP_MAX_NORM
            )

            self.optimizer_instance.step()

            batch_elapsed_time = time.time() - batch_timer_start
            cumulative_epoch_loss += computed_loss.item()

            if batch_index % 10 == 0:
                self.performance_tracker.log_training_batch(epoch_number, batch_index, computed_loss.item(), batch_elapsed_time)

        average_epoch_loss = cumulative_epoch_loss / len(self.training_dataloader)
        return average_epoch_loss

    def validate_model_performance(self) -> float:
        self.detector_model.get_model_instance().eval()
        cumulative_validation_loss = 0

        with torch.no_grad():
            for batch_content in self.validation_dataloader:
                image_tensor = batch_content['image'].to(self.device)
                bbox_tensor = batch_content['bboxes'].to(self.device)

                predicted_bboxes = self.detector_model.get_model_instance()(image_tensor)
                computed_loss = self.bbox_loss_function(predicted_bboxes, bbox_tensor)
                cumulative_validation_loss += computed_loss.item()

        average_validation_loss = cumulative_validation_loss / len(self.validation_dataloader)

        if average_validation_loss < self.best_validation_loss_value:
            self.best_validation_loss_value = average_validation_loss
            self.patience_counter = 0
            self.detector_model.save_model_checkpoint(f'{CHECKPOINT_DIR}/best_localization_model.pt')
        else:
            self.patience_counter += 1

        return average_validation_loss

    def execute_training(self, total_epochs: int = EPOCHS):
        for epoch_index in range(total_epochs):
            self.performance_tracker.start_training_epoch(epoch_index)

            if epoch_index == 10:
                self.detector_model.unfreeze_layers_progressively(unfreeze_fraction=0.5)
                for optimizer_param_group in self.optimizer_instance.param_groups:
                    optimizer_param_group['lr'] *= 0.1

            elif epoch_index == 30:
                self.detector_model.unfreeze_layers_progressively(unfreeze_fraction=1.0)
                for optimizer_param_group in self.optimizer_instance.param_groups:
                    optimizer_param_group['lr'] *= 0.1

            epoch_training_loss = self.train_single_epoch(epoch_index)
            epoch_validation_loss = self.validate_model_performance()

            self.scheduler_instance.step(epoch_validation_loss)

            self.performance_tracker.end_training_epoch(epoch_index, epoch_training_loss, epoch_validation_loss)

            print(f"Epoch {epoch_index+1:3d}/{total_epochs} | Train Loss: {epoch_training_loss:.4f} | Val Loss: {epoch_validation_loss:.4f} | "
                  f"Patience: {self.patience_counter}/{self.patience_threshold}")

            if self.patience_counter >= self.patience_threshold:
                print("Early stopping triggered due to no improvement")
                break

        self.performance_tracker.export_to_json_file()
        print(self.performance_tracker.get_performance_summary())


def load_yaml_configuration(config_filepath: str) -> Dict:
    with open(config_filepath, 'r') as config_file:
        loaded_configuration = yaml.safe_load(config_file)
    return loaded_configuration


def load_image_and_bbox_data(data_directory: str) -> tuple:
    image_list = []
    bbox_list = []

    if Path(data_directory).name == "CarDD_SOD":
        # Scan CarDD-TR, CarDD-VAL, CarDD-TE folders
        subdirs = [d for d in listdir(data_directory) if Path(join(data_directory, d)).is_dir()]
        raw_directories = [join(data_directory, d) for d in subdirs]
    else:
        raw_directories = [join(data_directory, 'raw')]

    for raw_directory in raw_directories:
        if not Path(raw_directory).exists():
            print(f"Warning: Directory {raw_directory} does not exist, skipping...")
            continue

        print(f"Loading from: {raw_directory}")
        for filename in listdir(raw_directory):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = join(raw_directory, filename)
                bbox_path = join(raw_directory, filename.replace(filename.split('.')[-1], 'txt'))
                
                if isfile(bbox_path):
                    image_array = cv2.imread(image_path)
                    if image_array is not None:
                        with open(bbox_path, 'r') as bbox_file:
                            bboxes = [list(map(float, line.strip().split())) for line in bbox_file if line.strip()]
                        if len(bboxes) > 0:
                            image_list.append(image_array)
                            bbox_list.append(bboxes)

    if len(image_list) == 0:
        raise ValueError(f"No images found in {data_directory}. Ensure image files (.jpg/.png) have corresponding .txt bbox files")

    print(f"✓ Loaded {len(image_list)} images with bounding boxes")
    return image_list, bbox_list


def main():
    import argparse

    argument_parser = argparse.ArgumentParser(description='Train Stage 2: Damage Localization Model')
    argument_parser.add_argument('--config', type=str, default='config/modelconfig.yaml',
                        help='Path to configuration file')
    argument_parser.add_argument('--data', type=str, default=None,
                        help='Path to your dataset (e.g., C:\\path\\to\\CarDD_SOD)')
    argument_parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs to train')
    parsed_arguments = argument_parser.parse_args()

    data_path = parsed_arguments.data if parsed_arguments.data else DATA_DIR
    print(f"Using dataset path: {data_path}")

    try:
        loaded_config = load_yaml_configuration(parsed_arguments.config)
    except:
        loaded_config = {'training': {}}
    
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    image_collection, bbox_collection = load_image_and_bbox_data(data_path)

    image_preprocessor = ImagePreprocessor(target_size=INPUT_SIZE)
    data_augmentor = DataAugmentor(augmentation_settings=AUGMENTATION_CONFIG)

    data_splits = DatasetSplitter.divide_into_splits(image_collection, bbox_collection)
    
    dataloaders = DatasetSplitter.build_dataloaders(
        training_data_split=data_splits['train'],
        validation_data_split=data_splits['val'],
        testing_data_split=data_splits['test'],
        batch_size_value=loaded_config['training'].get('batch_size', BATCH_SIZE),
        image_preprocessor=image_preprocessor,
        data_augmentor=data_augmentor
    )

    damage_detector = DamageLocalizationModel(
        model_version=loaded_config['training'].get('model_version', 'm'),
        use_pretrained_weights=loaded_config['training'].get('pretrained', True)
    )
    damage_detector.freeze_early_layers(freeze_fraction=0.75)

    trainer = DamageLocalizationTrainer(
        detector_model=damage_detector,
        training_config=loaded_config['training'],
        training_dataloader=dataloaders['train'],
        validation_dataloader=dataloaders['val']
    )

    print("="*60)
    print("Stage 2 Trainer initialized successfully")
    print(f"Training dataset: {len(dataloaders['train'].dataset)} images")
    print(f"Validation dataset: {len(dataloaders['val'].dataset)} images")
    print(f"Starting training for {parsed_arguments.epochs} epochs...")
    print("="*60)

    trainer.execute_training(total_epochs=parsed_arguments.epochs)

    print(f"\n{'='*60}")
    print(f"✓ Training completed!")
    print(f"✓ Best model saved to: {CHECKPOINT_DIR}/best_localization_model.pt")
    print(f"✓ Performance logs saved to: {LOG_DIR}/performance_logs.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
