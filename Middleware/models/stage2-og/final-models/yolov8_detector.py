import torch
from ultralytics import YOLO
from pathlib import Path
from config.constants import MODEL_NAME, PRETRAINED, INPUT_SIZE, NUM_CLASSES

class DamageLocalizationModel:
    def __init__(self, model_version: str = "m", use_pretrained_weights: bool = PRETRAINED):
        self.model_version = model_version
        self.use_pretrained_weights = use_pretrained_weights

        model_identifier = f'yolov8{model_version}.pt' if use_pretrained_weights else f'yolov8{model_version}.yaml'
        self.detection_network = YOLO(model_identifier)

        self.frozen_layer_count = 0
        self.trainable_layer_count = 0
        self._calculate_layer_counts()

    def _calculate_layer_counts(self):
        for parameter in self.detection_network.model.parameters():
            if parameter.requires_grad:
                self.trainable_layer_count += 1

    def freeze_early_layers(self, freeze_fraction: float = 0.75):
        total_layer_count = len(list(self.detection_network.model.model))
        freeze_until_layer = int(total_layer_count * freeze_fraction)

        for layer_module in self.detection_network.model.model[:freeze_until_layer]:
            for parameter in layer_module.parameters():
                parameter.requires_grad = False

        self.frozen_layer_count = freeze_until_layer

    def unfreeze_layers_progressively(self, unfreeze_fraction: float = 0.5):
        total_layer_count = len(list(self.detection_network.model.model))
        unfreeze_from_layer = int(total_layer_count * (1 - unfreeze_fraction))

        for layer_module in self.detection_network.model.model[unfreeze_from_layer:]:
            for parameter in layer_module.parameters():
                parameter.requires_grad = True

        self.frozen_layer_count = unfreeze_from_layer

    def get_freezing_status(self) -> dict:
        frozen_parameter_count = sum(1 for p in self.detection_network.model.parameters() if not p.requires_grad)
        trainable_parameter_count = sum(1 for p in self.detection_network.model.parameters() if p.requires_grad)
        total_parameter_count = frozen_parameter_count + trainable_parameter_count

        return {
            'frozen': frozen_parameter_count,
            'trainable': trainable_parameter_count,
            'total': total_parameter_count,
            'trainable_percentage': (trainable_parameter_count / total_parameter_count) * 100 if total_parameter_count > 0 else 0
        }

    def set_learning_rate_value(self, learning_rate_value: float):
        self.detection_network.lr = learning_rate_value

    def get_model_instance(self):
        return self.detection_network

    def save_model_checkpoint(self, checkpoint_filepath: str):
        self.detection_network.save(checkpoint_filepath)

    def load_model_checkpoint(self, checkpoint_filepath: str):
        self.detection_network = YOLO(checkpoint_filepath)

    def get_model_information(self) -> dict:
        return {
            'model_name': MODEL_NAME,
            'input_size': INPUT_SIZE,
            'num_classes': NUM_CLASSES,
            'total_parameters': sum(p.numel() for p in self.detection_network.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.detection_network.model.parameters() if p.requires_grad),
        }
