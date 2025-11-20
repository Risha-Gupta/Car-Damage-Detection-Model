import os
import cv2
import torch
from detectron2.config import CfgNode
from detectron2.modeling import build_model
from typing import Dict


class DetectronModelService:
    """Detectron2 model service for inference using trained checkpoint (CPU-only)."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DetectronModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.model = None
        self.config = None
        self.confidence_threshold = 0.5
        self._load_model_checkpoint()
        self._initialized = True
        print("✅ Detectron2 Model loaded successfully (singleton, CPU mode)")

    def _load_model_checkpoint(self):
        """Load Detectron2 model from checkpoint using CPU only."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(
            os.path.join(base_dir, "../../../Middleware/models/stage2/models/model_checkpoint.pt")
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        # Allow Detectron2 config class to deserialize safely
        torch.serialization.add_safe_globals([CfgNode])

        # Force CPU-only loading
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Extract config and weights
        self.config = checkpoint.get("model_cfg") or checkpoint.get("config")
        if self.config is None:
            raise ValueError("Config not found in checkpoint")

        # ✅ Force CPU execution no matter what's inside the checkpoint
        if hasattr(self.config, "defrost"):
            self.config.defrost()
        self.config.MODEL.DEVICE = "cpu"
        if hasattr(self.config, "freeze"):
            self.config.freeze()

        model_state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint.get("state_dict")
        if model_state_dict is None:
            raise ValueError("Model weights not found in checkpoint")

        # Build and load model strictly on CPU
        self.model = build_model(self.config).to("cpu")
        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False)
        self.model.eval()

        print(f"✅ Detectron2 model checkpoint loaded from: {model_path}")
        if missing:
            print(f"⚠️ Missing keys: {len(missing)}")
        if unexpected:
            print(f"⚠️ Unexpected keys: {len(unexpected)}")

    def predict_damage_detectron(self, image_path: str) -> Dict:
        """
        Run Detectron2 inference on CPU.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing:
            - boxes: List of bounding boxes [x1, y1, x2, y2]
            - scores: Confidence scores (>= threshold)
            - classes: Predicted class IDs
            - num_detections: Total number of detections
        """
        if self.model is None:
            raise RuntimeError("Detectron2 model not loaded")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")

        # Prepare input tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1)  # HWC → CHW

        height, width = img.shape[:2]
        inputs = [{"image": img_tensor, "height": height, "width": width}]

        with torch.no_grad():
            outputs = self.model.inference(inputs)[0]

        instances = outputs.get("instances")
        if instances is None:
            return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

        scores = instances.scores.cpu().numpy()
        mask = scores >= self.confidence_threshold

        return {
            "boxes": instances.pred_boxes.tensor[mask].cpu().numpy().tolist(),
            "scores": scores[mask].tolist(),
            "classes": instances.pred_classes[mask].cpu().numpy().tolist(),
            "num_detections": int(mask.sum())
        }
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold for detections.
        
        Args:
            threshold: Float between 0 and 1
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.confidence_threshold = threshold
        print(f"✅ Confidence threshold updated to: {threshold}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model configuration details
        """
        if self.config is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "device": "cpu",
            "confidence_threshold": self.confidence_threshold,
            "model_type": self.config.MODEL.META_ARCHITECTURE if hasattr(self.config.MODEL, "META_ARCHITECTURE") else "unknown",
            "num_classes": self.config.MODEL.ROI_HEADS.NUM_CLASSES if hasattr(self.config.MODEL, "ROI_HEADS") else "unknown"
        }