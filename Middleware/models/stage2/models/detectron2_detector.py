import torch
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.config import config as detectron2_config

from config.constants import MODEL_NAME, INPUT_SIZE, NUM_CLASSES, METRICS_DECIMAL_PLACES
torch.serialization.add_safe_globals([detectron2_config.CfgNode])
class DamageLocalizationModel:
    """
    Detectron2-based damage localization model using Faster R-CNN with ResNet50 backbone.
    Supports transfer learning, dropout, and L2 regularization for improved generalization.
    """
    
    def __init__(self, model_version: str = "ResNet50", use_pretrained_weights: bool = True):
        self.model_version = model_version
        self.use_pretrained_weights = use_pretrained_weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.cfg = self._configure_model()
        self.model = build_model(self.cfg)
        self.model.to(self.device)
        
        self.frozen_layer_count = 0
        self.trainable_layer_count = 0
        self._calculate_layer_counts()

    def _configure_model(self) -> object:
        """Configure Detectron2 model with Faster R-CNN for COCO dataset."""
        cfg = get_cfg()
        
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        ))
        
        if self.use_pretrained_weights:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            )
        
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.MAX_ITER = 1000
        cfg.SOLVER.STEPS = (700,)
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.IMS_PER_BATCH = 16
        cfg.SOLVER.WEIGHT_DECAY = 0.001  # L2 regularization
        
        cfg.MODEL.ROI_HEADS.DROPOUT_RATE = 0.3
        cfg.MODEL.BACKBONE.FREEZE_AT = 2  # Freeze early backbone layers (transfer learning)
        
        cfg.MODEL.DEVICE = self.device
        cfg.INPUT.MIN_SIZE_TRAIN = INPUT_SIZE[0]
        cfg.INPUT.MAX_SIZE_TRAIN = INPUT_SIZE[1]
        cfg.INPUT.MIN_SIZE_TEST = INPUT_SIZE[0]
        cfg.INPUT.MAX_SIZE_TEST = INPUT_SIZE[1]
        
        return cfg

    def _calculate_layer_counts(self):
        """Count frozen and trainable parameters."""
        trainable_params = 0
        frozen_params = 0
        
        for param in self.model.parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        self.trainable_layer_count = trainable_params
        self.frozen_layer_count = frozen_params

    def freeze_early_layers(self, freeze_fraction: float = 0.75):
        """Freeze early layers for transfer learning."""
        backbone = self.model.backbone
        
        # Freeze backbone layers
        for name, param in backbone.named_parameters():
            depth = name.count('.')
            if depth < int(freeze_fraction * 10):
                param.requires_grad = False
                self.frozen_layer_count += param.numel()
        
        self._calculate_layer_counts()

    def unfreeze_layers_progressively(self, unfreeze_fraction: float = 0.5):
        """Unfreeze layers progressively during training."""
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                depth = name.count('.')
                if depth >= int((1 - unfreeze_fraction) * 10):
                    param.requires_grad = True
        
        self._calculate_layer_counts()

    def get_freezing_status(self) -> dict:
        """Return current freezing status."""
        frozen_parameter_count = sum(1 for p in self.model.parameters() if not p.requires_grad)
        trainable_parameter_count = sum(1 for p in self.model.parameters() if p.requires_grad)
        total_parameter_count = frozen_parameter_count + trainable_parameter_count
        
        return {
            'frozen': self.frozen_layer_count,
            'trainable': self.trainable_layer_count,
            'total': self.frozen_layer_count + self.trainable_layer_count,
            'trainable_percentage': (self.trainable_layer_count / (self.frozen_layer_count + self.trainable_layer_count) * 100) 
                                  if (self.frozen_layer_count + self.trainable_layer_count) > 0 else 0
        }

    def set_learning_rate_value(self, learning_rate_value: float):
        """Update learning rate for solver."""
        self.cfg.SOLVER.BASE_LR = learning_rate_value

    def get_model_instance(self):
        """Return the underlying Detectron2 model."""
        return self.model

    def get_cfg_instance(self):
        """Return the Detectron2 config."""
        return self.cfg

    def save_model_checkpoint(self, checkpoint_filepath: str):
        """Save model checkpoint."""
        Path(checkpoint_filepath).parent.mkdir(parents=True, exist_ok=True)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.save(checkpoint_filepath)
        print(f"Model checkpoint saved to {checkpoint_filepath}")

    def load_model_checkpoint(self, checkpoint_filepath: str):
        """Load model checkpoint."""
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(checkpoint_filepath)
        print(f"Model checkpoint loaded from {checkpoint_filepath}")

    def get_model_information(self) -> dict:
        """Return model information."""
        return {
            'model_name': MODEL_NAME,
            'framework': 'Detectron2',
            'backbone': 'ResNet50',
            'input_size': INPUT_SIZE,
            'num_classes': NUM_CLASSES,
            'frozen_parameters': self.frozen_layer_count,
            'trainable_parameters': self.trainable_layer_count,
            'total_parameters': self.frozen_layer_count + self.trainable_layer_count,
            'device': self.device
        }
