import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import io
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

class Stage2DamageLocalizationDetector:
    def __init__(self, model_path: str):
        self.detection_network = YOLO(model_path)
        self.target_image_size = (224, 224)
        self.detection_confidence_threshold = 0.5

    def preprocess_image_for_detection(self, image_array: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image_array, self.target_image_size, interpolation=cv2.INTER_LINEAR)
        normalized_image = resized_image.astype(np.float32) / 255.0
        return normalized_image

    def run_damage_detection(self, image_array: np.ndarray) -> dict:
        preprocessed_image = self.preprocess_image_for_detection(image_array)
        detection_results = self.detection_network.predict(
            source=preprocessed_image,
            conf=self.detection_confidence_threshold,
            verbose=False
        )
        
        return self.extract_detection_results(detection_results, image_array.shape)

    def extract_detection_results(self, detection_results, original_image_shape: tuple) -> dict:
        detected_damage_bboxes = []
        extracted_damage_regions = []
        
        if len(detection_results) == 0 or detection_results[0].boxes is None:
            return {
                'detected_damage_bboxes': [],
                'extracted_damage_regions': [],
                'damage_regions_count': 0
            }
        
        result_object = detection_results[0]
        
        for box_tensor in result_object.boxes:
            bbox_coordinates = box_tensor.xyxy[0].cpu().numpy()
            confidence_score = float(box_tensor.conf[0].cpu().numpy())
            
            x_min = int(bbox_coordinates[0])
            y_min = int(bbox_coordinates[1])
            x_max = int(bbox_coordinates[2])
            y_max = int(bbox_coordinates[3])
            
            detected_damage_bboxes.append({
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'confidence': round(confidence_score, 4),
                'bbox_area': (x_max - x_min) * (y_max - y_min)
            })
        
        return {
            'detected_damage_bboxes': detected_damage_bboxes,
            'extracted_damage_regions': extracted_damage_regions,
            'damage_regions_count': len(detected_damage_bboxes)
        }

    def localize_damages_in_image(self, image_array: np.ndarray) -> dict:
        detection_output = self.run_damage_detection(image_array)
        
        return {
            'stage2_status': 'completed',
            'detected_damage_bboxes': detection_output['detected_damage_bboxes'],
            'damage_regions_count': detection_output['damage_regions_count']
        }


class LocationService:
    def __init__(self):
        model_path = os.getenv('STAGE2_MODEL_PATH', 'Middleware/models/stage2/yoloBest.pt')
        self.detector = Stage2DamageLocalizationDetector(model_path)

    def predict(self, img_bytes: bytes) -> dict:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_arr = np.array(img)
            
            result = self.detector.localize_damages_in_image(img_arr)
            
            return result
        
        except Exception as e:
            raise Exception(f"Stage 2 localization error: {str(e)}")
