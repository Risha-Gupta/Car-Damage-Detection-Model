import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import time
from pathlib import Path
from typing import Dict, Tuple

from models.yolov8_detector import YOLOv8Detector
from data_pipeline.preprocessing import ImagePreprocessor, BboxProcessor
from data_pipeline.augmentation import DataAugmentor
from data_pipeline.loader import DataSplitter
from logging.performance_logger import PerformanceLogger
from logging.excel_reporter import ExcelReporter
from config.constants import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_MAX_NORM
)

class Stage2Trainer:
    def __init__(self, model: YOLOv8Detector, config: Dict,
                 train_loader, validation_loader):
        self.model = model
        self.configuration = config
        self.training_data_loader = train_loader
        self.validation_data_loader = validation_loader

        self.loss_function = nn.SmoothL1Loss()

        self.optimizer = Adam(
            self.model.get_model().parameters(),
            lr=config.get('learning_rate', LEARNING_RATE),
            weight_decay=config.get('weight_decay', WEIGHT_DECAY)
        )

        self.learning_rate_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.1),
            patience=config.get('scheduler_patience', 3),
            min_lr=config.get('min_lr', 1e-5)
        )

        self.best_validation_loss = float('inf')
        self.early_stopping_threshold = config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
        self.early_stopping_counter = 0

        self.performance_tracker = PerformanceLogger()
        self.result_reporter = ExcelReporter()

    def train_epoch(self, epoch_number: int) -> float:
        self.model.get_model().train()
        cumulative_loss = 0

        for batch_index, batch_data in enumerate(self.training_data_loader):
            image_batch = batch_data['image'].to('cuda' if torch.cuda.is_available() else 'cpu')
            bbox_batch = batch_data['bboxes'].to('cuda' if torch.cuda.is_available() else 'cpu')

            batch_start_time = time.time()

            predicted_output = self.model.get_model()(image_batch)

            computed_loss = self.loss_function(predicted_output, bbox_batch)

            self.optimizer.zero_grad()
            computed_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.get_model().parameters(),
                max_norm=GRADIENT_CLIP_MAX_NORM
            )

            self.optimizer.step()

            batch_elapsed_time = time.time() - batch_start_time
            cumulative_loss += computed_loss.item()

            if batch_index % 10 == 0:
                self.performance_tracker.log_training_batch(epoch_number, batch_index, computed_loss.item(), batch_elapsed_time)

        average_epoch_loss = cumulative_loss / len(self.training_data_loader)
        return average_epoch_loss

    def validate(self) -> float:
        self.model.get_model().eval()
        cumulative_validation_loss = 0

        with torch.no_grad():
            for batch_data in self.validation_data_loader:
                image_batch = batch_data['image'].to('cuda' if torch.cuda.is_available() else 'cpu')
                bbox_batch = batch_data['bboxes'].to('cuda' if torch.cuda.is_available() else 'cpu')

                predicted_output = self.model.get_model()(image_batch)
                computed_loss = self.loss_function(predicted_output, bbox_batch)
                cumulative_validation_loss += computed_loss.item()

        average_validation_loss = cumulative_validation_loss / len(self.validation_data_loader)

        if average_validation_loss < self.best_validation_loss:
            self.best_validation_loss = average_validation_loss
            self.early_stopping_counter = 0
            self.model.save_checkpoint('models/checkpoints/best_model.pt')
        else:
            self.early_stopping_counter += 1

        return average_validation_loss

    def train(self, epochs: int = EPOCHS):
        for epoch_index in range(epochs):
            self.performance_tracker.start_training_epoch(epoch_index)

            if epoch_index == 10:
                self.model.unfreeze_backbone(unfreeze_percentage=0.5)
                for optimizer_group in self.optimizer.param_groups:
                    optimizer_group['lr'] *= 0.1

            elif epoch_index == 30:
                self.model.unfreeze_backbone(unfreeze_percentage=1.0)
                for optimizer_group in self.optimizer.param_groups:
                    optimizer_group['lr'] *= 0.1

            epoch_training_loss = self.train_epoch(epoch_index)

            epoch_validation_loss = self.validate()

            self.learning_rate_scheduler.step(epoch_validation_loss)

            self.performance_tracker.end_training_epoch(epoch_index, epoch_training_loss, epoch_validation_loss)

            print(f"Epoch {epoch_index+1:3d}/{epochs} | Train Loss: {epoch_training_loss:.4f} | Val Loss: {epoch_validation_loss:.4f} | "
                  f"Patience: {self.early_stopping_counter}/{self.early_stopping_threshold}")

            if self.early_stopping_counter >= self.early_stopping_threshold:
                break

        self.performance_tracker.export_json()
        print(self.performance_tracker.get_performance_summary())


def load_config(config_file_path: str) -> Dict:
    with open(config_file_path, 'r') as config_file:
        configuration = yaml.safe_load(config_file)
    return configuration


def main():
    import argparse

    argument_parser = argparse.ArgumentParser(description='Train Stage 2 model')
    argument_parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to config file')
    parsed_arguments = argument_parser.parse_args()

    configuration = load_config(parsed_arguments.config)

    detector_model = YOLOv8Detector(model_size='m', pretrained=True)
    detector_model.freeze_backbone(freeze_percentage=0.75)


if __name__ == '__main__':
    main()
