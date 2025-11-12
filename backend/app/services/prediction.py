import numpy as np
from PIL import Image
import io, os, cv2
import tensorflow as tf
import keras
from keras.layers import Layer
from typing import Dict

import torch
from detectron2.config import CfgNode
from detectron2.modeling import build_model

class ChannelSlice(Layer):
    """Custom layer to slice channels - serializes properly across Python versions"""
    def __init__(self, start_channel, end_channel, **kwargs):
        super().__init__(**kwargs)
        self.start_channel = start_channel
        self.end_channel = end_channel
    
    def call(self, inputs):
        return inputs[:, :, :, self.start_channel:self.end_channel]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "start_channel": self.start_channel,
            "end_channel": self.end_channel
        })
        return config

class DetectronModelService:
    """
    Detectron2 model service for inference using trained checkpoint.
    Handles PyTorch 2.6+ serialization with weights_only=False.
    Singleton pattern ensures model is loaded only once at startup.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DetectronModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.config = None
        self.confidence_threshold = 0.5
        self._load_model_checkpoint()
        self._initialized = True
        print("✅ Detectron2 Model loaded successfully (singleton)")
    
    def _load_model_checkpoint(self):
        """Load Detectron2 model from checkpoint with PyTorch 2.6+ compatibility"""
        try:
            # Get model path from environment or use default
            model_path = os.getenv("MODEL_PATH", "./models/model_checkpoint.pt")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            
            torch.serialization.add_safe_globals([CfgNode])
            
            # Load checkpoint with weights_only=False for full compatibility
            checkpoint = torch.load(model_path, weights_only=False)
            
            # Extract config and model state dict
            self.config = checkpoint.get("model_cfg", checkpoint.get("config", None))
            if self.config is None:
                raise ValueError("Config not found in checkpoint")
            
            model_state_dict = checkpoint.get("model", checkpoint.get("state_dict", None))
            if model_state_dict is None:
                raise ValueError("Model state dict not found in checkpoint")
            
            # Build and load model
            self.model = build_model(self.config)
            self.model.load_state_dict(model_state_dict)
            self.model.eval()
            
            print(f"✅ Detectron2 model checkpoint loaded from: {model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load Detectron2 checkpoint: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def predict_damage_detectron(self, image_path: str) -> Dict:
        """
        Run Detectron2 inference on image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with boxes, scores, classes, num_detections
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Read and prepare image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image at {image_path}")
            
            # Convert BGR to RGB for model
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to torch tensor in CHW format
            img_tensor = torch.from_numpy(img_rgb).float()
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
            
            # Prepare input for model
            height, width = img.shape[:2]
            input_dict = {
                "image": img_tensor,
                "height": height,
                "width": width
            }
            
            # Run inference
            with torch.no_grad():
                predictions = self.model.inference([input_dict])[0]
            
            # Extract and filter predictions by confidence threshold
            instances = predictions.get("instances", None)
            if instances is None:
                return {
                    "boxes": [],
                    "scores": [],
                    "classes": [],
                    "num_detections": 0
                }
            
            # Filter by confidence threshold
            scores = instances.scores.cpu().numpy()
            confidence_mask = scores >= self.confidence_threshold
            
            boxes = instances.pred_boxes.tensor[confidence_mask].cpu().numpy().tolist()
            classes = instances.pred_classes[confidence_mask].cpu().numpy().tolist()
            filtered_scores = scores[confidence_mask].tolist()
            
            return {
                "boxes": boxes,
                "scores": filtered_scores,
                "classes": classes,
                "num_detections": len(boxes)
            }
            
        except Exception as e:
            print(f"❌ Inference error: {e}")
            raise RuntimeError(f"Inference failed: {e}")

class PredictionService:
    """
    Stage 1: Binary damage classification (is_damaged or not_damaged)
    Input: raw image bytes
    Output: damage probability and classification
    """
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "../../../Middleware/models/stage1/mobilenetv3_canny_final.keras")
        self.threshold = 0.5
        self.img_size = 224
        self.model = self._load_model()
        print(f"✅ Stage 1 Model loaded successfully from: {self.model_path}")

    def _load_model(self):
        """Load trained model"""
        try:
            model = keras.models.load_model(
                self.model_path,
                custom_objects={'ChannelSlice': ChannelSlice}
            )
            return model
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model from {self.model_path}: {e}")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image: resize, convert to RGB, apply Canny edges"""
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((224, 224))
        img_rgb = np.array(image, dtype=np.uint8)

        # Apply Canny edge detection
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Normalize and combine: RGB + edge channels
        img_rgb = img_rgb / 255.0
        edges = edges / 255.0
        combined = np.concatenate([img_rgb, edges], axis=-1)  # (224, 224, 6)

        # Add batch dimension
        combined = np.expand_dims(combined, axis=0).astype(np.float32)
        return combined

    def predict(self, image_bytes: bytes) -> Dict[str, any]:
        """
        Classify if image contains damage.
        Returns: damage probability and binary classification
        """
        processed_image = self.preprocess_image(image_bytes)
        preds = self.model.predict(processed_image, verbose=0)
        prediction = float(preds[0][0])

        is_damaged = prediction > self.threshold

        return {
            "is_damaged": bool(is_damaged),
            "confidence": round(prediction, 4),
            "damage_probability": round(prediction, 4),
            "status": "damaged" if is_damaged else "not_damaged"
        }


class Stage4DamageClassifier:
    """
    Stage 4: Classify damage type and severity based on segmentation results
    Input: segmentation detections, original image, damage bboxes
    Output: damage classification, severity levels, risk assessment
    """
    def __init__(self):
        # Damage severity mapping based on detection area and count
        self.severity_levels = {
            "minor": {"min_area": 0, "max_area": 5000, "description": "Small, localized damage"},
            "moderate": {"min_area": 5000, "max_area": 20000, "description": "Medium-sized damage regions"},
            "severe": {"min_area": 20000, "max_area": float('inf'), "description": "Large or multiple damage areas"}
        }
        
        self.damage_types = {
            "scratch": "Surface-level damage, minimal structural impact",
            "dent": "Indentation without paint loss",
            "crack": "Fracture in paint or material",
            "burn": "Heat-related damage or discoloration",
            "rust": "Oxidation and corrosion",
            "other": "Unclassified damage"
        }

    def classify_damage(self, segmentation_data: dict, bbox_data: dict, image_array: np.ndarray) -> dict:
        """
        Classify damage based on segmentation and bounding box data.
        Returns severity, damage type estimates, and repair priority.
        """
        # Extract metrics from segmentation results
        detection_count = segmentation_data.get("detection_count", 0)
        detections = segmentation_data.get("detections", [])
        
        # Extract damage localization info
        damage_bboxes = bbox_data.get("detected_damage_bboxes", [])
        total_damage_area = sum([bbox.get("bbox_area", 0) for bbox in damage_bboxes])
        
        # Image dimensions for percentage calculation
        image_area = image_array.shape[0] * image_array.shape[1]
        damage_coverage_percent = (total_damage_area / image_area * 100) if image_area > 0 else 0
        
        # Determine severity level
        severity = self._assess_severity(total_damage_area, detection_count)
        
        # Estimate damage type from detection patterns
        damage_type_estimate = self._estimate_damage_type(detections, damage_coverage_percent)
        
        # Calculate repair priority
        repair_priority = self._calculate_priority(severity, damage_coverage_percent, detection_count)
        
        return {
            "damage_severity": severity,
            "damage_type": damage_type_estimate,
            "total_damage_area_pixels": int(total_damage_area),
            "damage_coverage_percent": round(damage_coverage_percent, 2),
            "detection_count": detection_count,
            "repair_priority": repair_priority,
            "classification_metadata": {
                "avg_confidence": round(np.mean([d.get("confidence", 0) for d in detections]), 4) if detections else 0,
                "severity_mapping": self.severity_levels[severity]
            }
        }

    def _assess_severity(self, total_area: int, detection_count: int) -> str:
        """Determine severity: minor, moderate, or severe"""
        # Weight: area (60%), detection count (40%)
        if total_area <= 5000 and detection_count <= 2:
            return "minor"
        elif total_area <= 20000 and detection_count <= 5:
            return "moderate"
        else:
            return "severe"

    def _estimate_damage_type(self, detections: list, coverage: float) -> str:
        """Estimate damage type based on detection patterns"""
        if not detections:
            return "no_damage"
        
        # Simple heuristic: high coverage suggests dent/burn, lower coverage suggests scratch/crack
        if coverage > 10:
            return "dent"
        elif coverage > 3:
            return "crack"
        else:
            return "scratch"

    def _calculate_priority(self, severity: str, coverage: float, count: int) -> str:
        """Calculate repair priority: low, medium, high"""
        if severity == "severe" or coverage > 15:
            return "high"
        elif severity == "moderate" or coverage > 5:
            return "medium"
        else:
            return "low"
