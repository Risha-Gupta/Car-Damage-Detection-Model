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


# -------------------------------
# Custom Layer for Keras Model
# -------------------------------
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


# -------------------------------
# Detectron2 Model Service (CPU Only)
# -------------------------------
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

        # ✅ Force CPU execution no matter what’s inside the checkpoint
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
        """Run Detectron2 inference on CPU."""
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


# -------------------------------
# Stage 1: Binary Damage Classifier
# -------------------------------
class PredictionService:
    """Stage 1: Binary damage classification (is_damaged or not_damaged)."""
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(
            base_dir,
            "../../../Middleware/models/stage1/models/mobilenetv3_canny_final.keras"
        )
        self.threshold = 0.82
        self.img_size = 224
        self.model = self._load_model()

        print(f"Stage 1 Model loaded successfully from: {self.model_path}")

    def _load_model(self):
        try:
            return keras.models.load_model(self.model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {self.model_path}: {e}"
            )

    def preprocess_image(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((self.img_size, self.img_size))

        img_rgb = np.array(image, dtype=np.uint8)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 120)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        img_rgb = img_rgb.astype(np.float32) / 255.0
        edges = edges.astype(np.float32) / 255.0

        img_rgb = np.expand_dims(img_rgb, axis=0)
        edges = np.expand_dims(edges, axis=0)

        return img_rgb, edges

   
    def predict(self, image_bytes: bytes) -> Dict:
        img_rgb, edges = self.preprocess_image(image_bytes)

        prob = float(self.model.predict([img_rgb, edges], verbose=0)[0][0])
        is_damaged = prob > self.threshold

        return {
            "is_damaged": bool(is_damaged),
            "confidence": round(prob, 4),
            "damage_probability": round(prob, 4),
            "status": "damaged" if is_damaged else "not_damaged"
        }
