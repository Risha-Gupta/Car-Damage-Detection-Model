import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple
from config.constants import INPUT_SIZE

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = INPUT_SIZE):
        self.target_size = target_size

    def load_and_resize(self, image_path: str) -> np.ndarray:
        image_data = cv2.imread(image_path)
        if image_data is None:
            raise ValueError(f"Cannot load image from {image_path}")
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image_data, self.target_size, interpolation=cv2.INTER_LINEAR)
        return resized_image

    def apply_canny_edges(self, image_bgr: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2GRAY)
        edge_map = cv2.Canny(gray_image, 100, 200)
        edge_map_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
        return edge_map_rgb

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            normalized = image.astype(np.float32) / 255.0
        else:
            normalized = image.astype(np.float32)
        return normalized

    def create_six_channel_input(self, image_path: str) -> np.ndarray:
        rgb_image = self.load_and_resize(image_path)
        edge_image = self.apply_canny_edges(rgb_image)
        
        rgb_normalized = self.normalize_image(rgb_image)
        edge_normalized = self.normalize_image(edge_image)
        
        six_channel_input = np.concatenate([rgb_normalized, edge_normalized], axis=-1)
        return six_channel_input

    def validate_image(self, image: np.ndarray) -> bool:
        if image is None or len(image.shape) < 2:
            return False
        if image.shape[0] == 0 or image.shape[1] == 0:
            return False
        return True

    def process_for_inference(self, image: np.ndarray) -> np.ndarray:
        if not self.validate_image(image):
            raise ValueError("Invalid image provided")
        
        resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        edge_image = self.apply_canny_edges(resized_image)
        
        rgb_normalized = self.normalize_image(resized_image)
        edge_normalized = self.normalize_image(edge_image)
        
        six_channel_input = np.concatenate([rgb_normalized, edge_normalized], axis=-1)
        return six_channel_input

def process_single_image_simple(image_path: str, target_size: Tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    """
    Load and process a single image for inference.
    
    Args:
        image_path: Path to image file
        target_size: Target size tuple (height, width)
    
    Returns:
        Processed image as numpy array ready for model inference
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    return image

class BboxProcessor:
    def __init__(self, original_image_height: int, original_image_width: int, 
                 target_height: int, target_width: int):
        self.original_image_height = original_image_height
        self.original_image_width = original_image_width
        self.target_height = target_height
        self.target_width = target_width
        self.scale_horizontal = target_width / original_image_width
        self.scale_vertical = target_height / original_image_height

    def scale_single_bbox(self, bbox_coordinates: list) -> list:
        x_top_left, y_top_left, x_bottom_right, y_bottom_right = bbox_coordinates
        scaled_bbox = [
            int(x_top_left * self.scale_horizontal),
            int(y_top_left * self.scale_vertical),
            int(x_bottom_right * self.scale_horizontal),
            int(y_bottom_right * self.scale_vertical)
        ]
        return scaled_bbox

    def scale_multiple_bboxes(self, bbox_list: list) -> list:
        return [self.scale_single_bbox(bbox) for bbox in bbox_list]

    def crop_region_from_image(self, image: np.ndarray, bbox_coordinates: list) -> np.ndarray:
        x_top_left, y_top_left, x_bottom_right, y_bottom_right = bbox_coordinates
        cropped_region = image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
        return cropped_region
