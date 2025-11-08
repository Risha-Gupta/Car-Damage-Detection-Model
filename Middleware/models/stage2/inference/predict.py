import torch
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict
import tensorflow as tf
from models.yolov8_detector import DamageLocalizationModel
from data_pipeline.preprocessing import ImagePreprocessor, BboxProcessor
from logging.performance_logger import PerformanceLogger
from config.constants import STAGE1_MODEL_PATH, STAGE1_DAMAGE_THRESHOLD

class Stage1DamageClassifier:
    def __init__(self, model_checkpoint_path: str = STAGE1_MODEL_PATH):
        self.stage1_network = tf.keras.models.load_model(model_checkpoint_path)
        self.damage_classification_threshold = STAGE1_DAMAGE_THRESHOLD

    def classify_damage_status(self, car_image: np.ndarray) -> tuple:
        image_resized = tf.image.resize(car_image, (224, 224))
        image_normalized = image_resized / 255.0
        
        grayscale_image = tf.image.rgb_to_grayscale(image_resized)
        edge_detection = cv2.Canny(grayscale_image.numpy().astype(np.uint8) * 255, 100, 200)
        edges_as_rgb = cv2.cvtColor(edge_detection, cv2.COLOR_GRAY2RGB)
        edges_normalized = edges_as_rgb / 255.0
        
        six_channel_input = tf.concat([image_normalized, edges_normalized], axis=-1)
        six_channel_input = tf.expand_dims(six_channel_input, 0)
        
        damage_probability = self.stage1_network.predict(six_channel_input)[0][0]
        is_damaged_boolean = damage_probability >= self.damage_classification_threshold
        
        return is_damaged_boolean, float(damage_probability)


class DamageLocalizationPredictor:
    def __init__(self, localization_model_checkpoint: str = None):
        self.localization_network = DamageLocalizationModel(model_version='m', use_pretrained_weights=True)

        if localization_model_checkpoint and Path(localization_model_checkpoint).exists():
            self.localization_network.load_model_checkpoint(localization_model_checkpoint)

        self.image_processor_instance = ImagePreprocessor(target_size=(512, 512))
        self.timing_recorder = PerformanceLogger()
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict_damage_locations(self, car_image: np.ndarray, detection_confidence_threshold: float = 0.5) -> Dict:
        pipeline_timer_start = time.time()

        preprocessing_timer_start = time.time()
        processed_image_array = self.image_processor_instance.process(car_image)
        image_tensor_input = torch.from_numpy(processed_image_array).permute(2, 0, 1).unsqueeze(0).to(self.device_type)
        preprocessing_duration = time.time() - preprocessing_timer_start

        inference_timer_start = time.time()
        with torch.no_grad():
            model_output_predictions = self.localization_network.get_model_instance()(image_tensor_input)
        inference_duration = time.time() - inference_timer_start

        postprocessing_timer_start = time.time()
        detected_damage_bboxes = []
        detection_confidence_values = []

        if hasattr(model_output_predictions, 'boxes'):
            for detected_box_object in model_output_predictions.boxes:
                confidence_score = float(detected_box_object.conf[0])
                if confidence_score >= detection_confidence_threshold:
                    bbox_coordinates_list = detected_box_object.xyxy[0].cpu().numpy().astype(int).tolist()
                    detected_damage_bboxes.append(bbox_coordinates_list)
                    detection_confidence_values.append(confidence_score)

        postprocessing_duration = time.time() - postprocessing_timer_start
        pipeline_total_duration = time.time() - pipeline_timer_start

        return {
            'bboxes': detected_damage_bboxes,
            'confidence_scores': detection_confidence_values,
            'num_detections': len(detected_damage_bboxes),
            'execution_time': round(pipeline_total_duration, 4),
            'timing_breakdown': {
                'preprocessing': round(preprocessing_duration, 4),
                'inference': round(inference_duration, 4),
                'postprocessing': round(postprocessing_duration, 4)
            }
        }

    def predict_multiple_images(self, image_array_list: List[np.ndarray],
                               detection_confidence_threshold: float = 0.5) -> List[Dict]:
        batch_prediction_results = []
        for image_index, car_image in enumerate(image_array_list):
            single_prediction = self.predict_damage_locations(car_image, detection_confidence_threshold)
            batch_prediction_results.append(single_prediction)

        return batch_prediction_results


class EndToEndDamagePipeline:
    def __init__(self, stage1_classifier_instance: Stage1DamageClassifier = None, 
                 stage2_model_checkpoint_path: str = None):
        self.stage1_binary_classifier = stage1_classifier_instance or Stage1DamageClassifier()
        self.stage2_localization_predictor = DamageLocalizationPredictor(localization_model_checkpoint=stage2_model_checkpoint_path)
        self.full_pipeline_timer = PerformanceLogger()

    def process_single_car_image(self, car_image: np.ndarray, image_name_identifier: str = None) -> Dict:
        execution_timings_dict = {}

        stage1_timer_start = time.time()
        is_car_damaged, damage_probability_score = self.stage1_binary_classifier.classify_damage_status(car_image)
        execution_timings_dict['stage1'] = time.time() - stage1_timer_start

        if not is_car_damaged:
            return {
                'image_id': image_name_identifier,
                'is_damaged': False,
                'stage1_confidence': round(damage_probability_score, 4),
                'timings': execution_timings_dict
            }

        stage2_timer_start = time.time()
        localization_results_dict = self.stage2_localization_predictor.predict_damage_locations(car_image)
        execution_timings_dict['stage2'] = localization_results_dict['execution_time']

        self.full_pipeline_timer.log_full_pipeline(
            image_name_identifier or 'unknown',
            {
                'stage1': execution_timings_dict.get('stage1', 0),
                'stage2': execution_timings_dict.get('stage2', 0),
                'total': sum(execution_timings_dict.values())
            }
        )

        return {
            'image_id': image_name_identifier,
            'is_damaged': True,
            'stage1_confidence': round(damage_probability_score, 4),
            'stage2_bboxes': localization_results_dict['bboxes'],
            'stage2_confidence_scores': localization_results_dict['confidence_scores'],
            'num_damage_regions_detected': localization_results_dict['num_detections'],
            'timings': execution_timings_dict,
            'total_pipeline_time': sum(execution_timings_dict.values())
        }

    def process_multiple_car_images(self, car_image_list: List[np.ndarray],
                                   image_name_identifiers: List[str] = None) -> List[Dict]:
        if image_name_identifiers is None:
            image_name_identifiers = [f'image_{image_idx}' for image_idx in range(len(car_image_list))]

        all_results_list = []
        for car_image, image_name in zip(car_image_list, image_name_identifiers):
            result_dict = self.process_single_car_image(car_image, image_name)
            all_results_list.append(result_dict)

        return all_results_list
