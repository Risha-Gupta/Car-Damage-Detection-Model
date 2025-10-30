import numpy as np
from PIL import Image
import io
from typing import Dict
import tensorflow as tf
import keras
import os
class PredictionService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "../../../Middleware/models/stage1/resnet50_damage.keras")
        self.threshold = 0.5
        self.img_size = 224
        self.model = self._load_model()
        print(f"✅ Model loaded successfully from: {self.model_path}")
    
    def _load_model(self):
        """Load your trained model"""
        try:
            model = keras.models.load_model(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for model input"""
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model's expected input size (adjust as needed)
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_bytes: bytes) -> Dict[str, any]:
        """Make prediction on image"""
        # Preprocess image
        processed_image = self.preprocess_image(image_bytes)
        
        preds = self.model.predict(processed_image)
        prediction = float(preds[0][0])
        
        # Determine if damaged
        is_damaged = bool(prediction > self.threshold)
        confidence = float(prediction)
        
        return {
            "is_damaged": is_damaged,
            "confidence": confidence,
            "damage_probability": round(prediction, 4),
            "status": "damaged" if is_damaged else "not_damaged"
        }
