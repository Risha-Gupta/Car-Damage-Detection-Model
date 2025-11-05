import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
from config.constants import MODEL_NAME, PRETRAINED, INPUT_SIZE, NUM_CLASSES

class YOLOv8Detector:
    def __init__(self, model_size: str = "m", pretrained: bool = PRETRAINED):
        self.model_size = model_size
        self.pretrained = pretrained

        model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
        self.model = YOLO(model_name)

        self.frozen_layer_count = 0
        self.trainable_layer_count = 0
        self._count_layers()

    def _count_layers(self):
        for param in self.model.model.parameters():
            if param.requires_grad:
                self.trainable_layer_count += 1

    def freeze_backbone(self, freeze_percentage: float = 0.75):
        total_layers = len(list(self.model.model.model))
        freeze_until = int(total_layers * freeze_percentage)

        for layer in self.model.model.model[:freeze_until]:
            for param in layer.parameters():
                param.requires_grad = False

        self.frozen_layer_count = freeze_until

    def unfreeze_backbone(self, unfreeze_percentage: float = 0.5):
        total_layers = len(list(self.model.model.model))
        unfreeze_from = int(total_layers * (1 - unfreeze_percentage))

        for layer in self.model.model.model[unfreeze_from:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.frozen_layer_count = unfreeze_from

    def get_frozen_status(self) -> dict:
        frozen_param_count = sum(1 for p in self.model.model.parameters() if not p.requires_grad)
        trainable_param_count = sum(1 for p in self.model.model.parameters() if p.requires_grad)

        return {
            'frozen': frozen_param_count,
            'trainable': trainable_param_count,
            'total': frozen_param_count + trainable_param_count,
            'trainable_percentage': (trainable_param_count / (frozen_param_count + trainable_param_count)) * 100
        }

    def set_learning_rate(self, learning_rate: float):
        self.model.lr = learning_rate

    def get_model(self):
        return self.model

    def save_checkpoint(self, filepath: str):
        self.model.save(filepath)

    def load_checkpoint(self, filepath: str):
        self.model = YOLO(filepath)

    def get_architecture_info(self) -> dict:
        return {
            'model_name': MODEL_NAME,
            'input_size': INPUT_SIZE,
            'num_classes': NUM_CLASSES,
            'parameters': sum(p.numel() for p in self.model.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.model.parameters() if p.requires_grad),
        }


class ModelComparison:
    pass
