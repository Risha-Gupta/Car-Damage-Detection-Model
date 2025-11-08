import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import time
from pathlib import Path
from typing import Dict
from models.detectron2_detector import DamageLocalizationModel
from data_pipeline.preprocessor import ImagePreprocessor
from data_pipeline.augmentation import DataAugmentor
from data_pipeline.loader import DatasetSplitter
from project_logging.performance_logger import PerformanceLogger
from project_logging.excel_reporter import ExcelReporter
from config.constants import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, DATA_DIR, CHECKPOINT_DIR, LOG_DIR, 
    REPORT_OUTPUT, AUGMENTATION_CONFIG, INPUT_SIZE, METRICS_DECIMAL_PLACES
)


class FocalLoss(torch.nn.Module):
    """Focal loss for handling imbalanced damage detection classes."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        ce_loss = torch.nn.functional.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DamageLocalizationTrainer:
    """Complete rewrite for Detectron2 training with focal loss and early stopping."""
    
    def __init__(self, detector_model: DamageLocalizationModel, training_config: Dict,
                 train_annotation_file: str, val_annotation_file: str,
                 base_image_dir: str):
        self.detector_model = detector_model
        self.training_configuration = training_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Register COCO dataset with Detectron2
        self._register_coco_datasets(
            train_annotation_file, val_annotation_file,
            base_image_dir
        )
        
        # Configure Detectron2 trainer
        self.cfg = detector_model.get_cfg_instance()
        self.cfg.DATASETS.TRAIN = ("damage_train",)
        self.cfg.DATASETS.TEST = ("damage_val",)
        
        self.cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
        self.cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
        self.cfg.SOLVER.WEIGHT_DECAY = training_config.get('weight_decay', WEIGHT_DECAY)
        
        self.model = detector_model.get_model_instance()
        
        # Setup optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=training_config.get('learning_rate', LEARNING_RATE),
            weight_decay=training_config.get('weight_decay', WEIGHT_DECAY)
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=training_config.get('scheduler_factor', 0.1),
            patience=training_config.get('scheduler_patience', 3),
            min_lr=training_config.get('min_lr', 1e-5)
        )
        
        self.best_validation_loss = float('inf')
        self.patience_counter = 0
        self.patience_threshold = training_config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
        
        self.performance_tracker = PerformanceLogger()
        self.result_exporter = ExcelReporter()
        
        print(f"Using device: {self.device}")

    def _register_coco_datasets(self, train_ann_file: str, val_ann_file: str, image_dir: str):
        """Register COCO datasets with Detectron2."""
        from detectron2.data.datasets import register_coco_instances
        
        try:
            register_coco_instances(
                "damage_train", {},
                train_ann_file,
                image_dir
            )
            register_coco_instances(
                "damage_val", {},
                val_ann_file,
                image_dir
            )
            print("Successfully registered COCO datasets")
        except Exception as e:
            print(f"Dataset registration error: {e}")
            raise

    def train_single_epoch(self, epoch_number: int) -> float:
        """Train single epoch with focal loss and regularization."""
        self.model.train()
        cumulative_epoch_loss = 0.0
        batch_count = 0

        with EventStorage() as storage:
            for batch_idx in range(self.cfg.SOLVER.IMS_PER_BATCH):
                batch_timer_start = time.time()
                
                # Get batch from dataloader
                try:
                    loss_dict = self.model(batch_idx)
                    
                    if isinstance(loss_dict, dict):
                        total_loss = sum(loss_dict.values())
                    else:
                        total_loss = loss_dict
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0
                    )
                    
                    self.optimizer.step()
                    
                    batch_elapsed_time = time.time() - batch_timer_start
                    cumulative_epoch_loss += total_loss.item()
                    batch_count += 1
                    
                    if batch_idx % 10 == 0:
                        self.performance_tracker.log_training_batch(
                            epoch_number, batch_idx,
                            total_loss.item(),
                            batch_elapsed_time
                        )
                
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue

        average_epoch_loss = cumulative_epoch_loss / batch_count if batch_count > 0 else 0
        return round(average_epoch_loss, METRICS_DECIMAL_PLACES)

    def validate_model_performance(self) -> float:
        """Validation with Detectron2 evaluator."""
        self.model.eval()
        
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader
        
        val_loader = build_detection_test_loader(self.cfg, "damage_val")
        evaluator = COCOEvaluator("damage_val", self.cfg, False, output_dir=str(Path(LOG_DIR)))
        
        try:
            results = inference_on_dataset(self.model, val_loader, evaluator)
            val_loss = results.get('bbox', {}).get('AP', 0)
            
            if val_loss < self.best_validation_loss:
                self.best_validation_loss = val_loss
                self.patience_counter = 0
                self.detector_model.save_model_checkpoint(f'{CHECKPOINT_DIR}/best_localization_model.pth')
            else:
                self.patience_counter += 1
            
            return round(val_loss, METRICS_DECIMAL_PLACES)
        
        except Exception as e:
            print(f"Validation error: {e}")
            return 0.0

    def execute_training(self, total_epochs: int = EPOCHS):
        """Execute training with early stopping."""
        for epoch_index in range(total_epochs):
            self.performance_tracker.start_training_epoch(epoch_index)
            
            if epoch_index == int(total_epochs * 0.3):
                self.detector_model.unfreeze_layers_progressively(unfreeze_fraction=0.5)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print("Unfroze middle layers, reduced LR by 10x")
            
            elif epoch_index == int(total_epochs * 0.7):
                self.detector_model.unfreeze_layers_progressively(unfreeze_fraction=1.0)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print("Unfroze all layers, reduced LR by 10x")
            
            epoch_training_loss = self.train_single_epoch(epoch_index)
            epoch_validation_loss = self.validate_model_performance()
            
            self.scheduler.step(epoch_validation_loss)
            
            self.performance_tracker.end_training_epoch(
                epoch_index,
                epoch_training_loss,
                epoch_validation_loss
            )
            
            print(f"Epoch {epoch_index+1:3d}/{total_epochs} | Train Loss: {epoch_training_loss:.4f} | "
                  f"Val Loss: {epoch_validation_loss:.4f} | Patience: {self.patience_counter}/{self.patience_threshold}")
            
            if self.patience_counter >= self.patience_threshold:
                print("Early stopping triggered due to no improvement")
                break
        
        self.performance_tracker.export_to_json_file()
        print(self.performance_tracker.get_performance_summary())


def load_yaml_configuration(config_filepath: str) -> Dict:
    with open(config_filepath, 'r') as config_file:
        loaded_configuration = yaml.safe_load(config_file)
    return loaded_configuration


def main():
    import argparse
    
    argument_parser = argparse.ArgumentParser(description='Train Stage 2: Damage Localization with Detectron2')
    argument_parser.add_argument('--config', type=str, default='config/modelconfig.yaml',
                                help='Path to configuration file')
    argument_parser.add_argument('--data', type=str, default=None,
                                help='Path to CarDD_COCO dataset root')
    argument_parser.add_argument('--epochs', type=int, default=EPOCHS,
                                help='Number of epochs to train')
    parsed_arguments = argument_parser.parse_args()
    
    data_path = Path(parsed_arguments.data if parsed_arguments.data else DATA_DIR)
    train_ann_file = data_path / 'annotations' / 'instances_train2017.json'
    val_ann_file = data_path / 'annotations' / 'instances_val2017.json'
    image_dir = data_path
    
    print(f"Using dataset path: {data_path}")
    print(f"Train annotations: {train_ann_file}")
    print(f"Val annotations: {val_ann_file}")
    
    if not train_ann_file.exists():
        raise FileNotFoundError(f"Train annotations not found: {train_ann_file}")
    if not val_ann_file.exists():
        raise FileNotFoundError(f"Val annotations not found: {val_ann_file}")
    
    try:
        loaded_config = load_yaml_configuration(parsed_arguments.config)
    except:
        loaded_config = {'training': {}}
    
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    damage_detector = DamageLocalizationModel(
        model_version='ResNet50',
        use_pretrained_weights=loaded_config['training'].get('pretrained', True)
    )
    damage_detector.freeze_early_layers(freeze_fraction=0.75)
    
    # Initialize trainer
    trainer = DamageLocalizationTrainer(
        detector_model=damage_detector,
        training_config=loaded_config['training'],
        train_annotation_file=str(train_ann_file),
        val_annotation_file=str(val_ann_file),
        base_image_dir=str(image_dir)
    )
    
    print("="*60)
    print("Stage 2 Trainer initialized successfully")
    print(f"Starting training for {parsed_arguments.epochs} epochs...")
    print("="*60)
    
    trainer.execute_training(total_epochs=parsed_arguments.epochs)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best model saved to: {CHECKPOINT_DIR}/best_localization_model.pth")
    print(f"Performance logs saved to: {LOG_DIR}/performance_logs.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
