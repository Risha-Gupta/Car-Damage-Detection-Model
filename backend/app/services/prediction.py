import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

class Stage1DamageClassifier:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.target_image_size = (224, 224)
        self.confidence_threshold = 0.5

    def load_image_from_path(self, image_path: str) -> np.ndarray:
        image_data = cv2.imread(image_path)
        if image_data is None:
            raise ValueError(f"Cannot load image from {image_path}")
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        return image_data

    def apply_canny_edge_detection(self, rgb_image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        edge_map = cv2.Canny(gray_image, 100, 200)
        edge_map_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
        return edge_map_rgb

    def normalize_image_values(self, image_data: np.ndarray) -> np.ndarray:
        normalized_image = image_data.astype(np.float32) / 255.0
        return normalized_image

    def create_six_channel_tensor(self, image_path: str) -> tuple:
        original_image = self.load_image_from_path(image_path)
        resized_image = cv2.resize(original_image, self.target_image_size, interpolation=cv2.INTER_LINEAR)
        
        edge_image = self.apply_canny_edge_detection(resized_image)
        
        rgb_normalized = self.normalize_image_values(resized_image)
        edge_normalized = self.normalize_image_values(edge_image)
        
        six_channel_tensor = np.concatenate([rgb_normalized, edge_normalized], axis=-1)
        return np.expand_dims(six_channel_tensor, axis=0), original_image

    def classify_damage_status(self, image_path: str) -> dict:
        six_channel_input, original_image = self.create_six_channel_tensor(image_path)
        
        damage_prediction_output = self.model.predict(six_channel_input, verbose=0)
        damage_probability_score = float(damage_prediction_output[0][0])
        
        is_car_damaged = damage_probability_score >= self.confidence_threshold
        damage_classification = "Damaged" if is_car_damaged else "Not Damaged"
        
        return {
            'stage1_classification': damage_classification,
            'damage_probability_score': round(damage_probability_score, 4),
            'is_car_damaged': is_car_damaged,
            'original_image': original_image,
            'original_image_shape': original_image.shape
        }
