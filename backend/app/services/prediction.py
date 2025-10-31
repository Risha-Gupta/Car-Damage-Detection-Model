import numpy as np
from PIL import Image
import io, os, cv2
import tensorflow as tf
import keras
from typing import Dict

class PredictionService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "../../../Middleware/models/stage1/mobilenetv3_canny_final.keras")
        self.threshold = 0.5
        self.img_size = 224
        self.model = self._load_model()
        print(f"✅ Model loaded successfully from: {self.model_path}")

    def _load_model(self):
        """Load trained model"""
        try:
            keras.config.enable_unsafe_deserialization()
            model = keras.models.load_model(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model from {self.model_path}: {e}")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((224, 224))
        img_rgb = np.array(image, dtype=np.uint8)

        # ---- Apply Canny edges ----
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Normalize and combine
        img_rgb = img_rgb / 255.0
        edges = edges / 255.0
        combined = np.concatenate([img_rgb, edges], axis=-1)  # (224, 224, 6)

        # Batch dimension
        combined = np.expand_dims(combined, axis=0).astype(np.float32)
        return combined


    def predict(self, image_bytes: bytes) -> Dict[str, any]:
        """Make prediction on image"""
        processed_image = self.preprocess_image(image_bytes)
        preds = self.model.predict(processed_image)
        prediction = float(preds[0][0])

        is_damaged = prediction > self.threshold

        return {
            "is_damaged": bool(is_damaged),
            "confidence": round(prediction, 4),
            "damage_probability": round(prediction, 4),
            "status": "damaged" if is_damaged else "not_damaged"
        }
