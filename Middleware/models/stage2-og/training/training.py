import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import time
from pathlib import Path
from typing import Dict, Tuple
import tensorflow as tf

from models.yolov8_detector import DamageLocalizationModel
from data_pipeline.preprocessing import ImagePreprocessor, BboxProcessor
from data_pipeline.augmentation import DataAugmentor
from data_pipeline.loader import DatasetSplitter
from logging.performance_logger import PerformanceLogger
from logging.excel_reporter import ExcelReporter
from config.constants import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_MAX_NORM,
    STAGE1_MODEL_PATH, STAGE1_DAMAGE_THRESHOLD
)

class Stage1BinaryClassifier:
    def __init__(self, model_path: str = STAGE1_MODEL_PATH):
        self.stage1_model = tf.keras.models.load_model(model_path)
        self.damage_threshold = STAGE1_DAMAGE_THRESHOLD

    def predict_damage_binary(self, image_array: np.ndarray) -> tuple:
        image_resized = tf.image.resize(image_array, (224, 224))
        image_normalized = image_resized / 255.0
        
        gray_image = tf.image.rgb_to_grayscale(image_resized)
        edges = cv2.Canny(gray_image.numpy().astype(np.uint8) * 255, 100, 200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_normalized = edges_rgb / 255.0
        
        combined_input = tf.concat([image_normalized, edges_normalized], axis=-1)
        combined_input = tf.expand_dims(combined_input, 0)
        
        prediction_score = self.stage1_model.predict(combined_input)[0][0]
        is_damaged = prediction_score >= self.damage_threshold
        
        return is_damaged, float(prediction_score)


class DamageLocalizationTrainer:
    def __init__(self, detector_model: DamageLocalizationModel, training_config: Dict,
                 training_dataloader, validation_dataloader,
                 stage1_classifier: Stage1BinaryClassifier = None):
        self.detector_model = detector_model
        self.training_configuration = training_config
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.stage1_classifier = stage1_classifier

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

    def train_single_epoch(self, epoch_number: int) -> float:
        self.detector_model.get_model_instance().train()
        cumulative_epoch_loss = 0

        for batch_index, batch_content in enumerate(self.training_dataloader):
            image_tensor = batch_content['image'].to('cuda' if torch.cuda.is_available() else 'cpu')
            bbox_tensor = batch_content['bboxes'].to('cuda' if torch.cuda.is_available() else 'cpu')

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
                image_tensor = batch_content['image'].to('cuda' if torch.cuda.is_available() else 'cpu')
                bbox_tensor = batch_content['bboxes'].to('cuda' if torch.cuda.is_available() else 'cpu')

                predicted_bboxes = self.detector_model.get_model_instance()(image_tensor)
                computed_loss = self.bbox_loss_function(predicted_bboxes, bbox_tensor)
                cumulative_validation_loss += computed_loss.item()

        average_validation_loss = cumulative_validation_loss / len(self.validation_dataloader)

        if average_validation_loss < self.best_validation_loss_value:
            self.best_validation_loss_value = average_validation_loss
            self.patience_counter = 0
            self.detector_model.save_model_checkpoint('models/checkpoints/best_localization_model.pt')
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

        self.performance_tracker.export_json()
        print(self.performance_tracker.get_performance_summary())


def load_yaml_configuration(config_filepath: str) -> Dict:
    with open(config_filepath, 'r') as config_file:
        loaded_configuration = yaml.safe_load(config_file)
    return loaded_configuration


def main():
    import argparse
    import numpy as np
    import cv2

    argument_parser = argparse.ArgumentParser(description='Train Stage 2: Damage Localization Model')
    argument_parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to configuration file')
    parsed_arguments = argument_parser.parse_args()

    loaded_config = load_yaml_configuration(parsed_arguments.config)

    damage_detector = DamageLocalizationModel(model_version='m', use_pretrained_weights=True)
    damage_detector.freeze_early_layers(freeze_fraction=0.75)

    stage1_damage_classifier = Stage1BinaryClassifier(model_path=STAGE1_MODEL_PATH)

    print("Stage 2 Trainer initialized successfully")


if __name__ == '__main__':
    main()
