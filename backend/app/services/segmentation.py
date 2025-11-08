import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import io
from PIL import Image
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Configure PyTorch to allow YOLO model loading (PyTorch 2.6+)
torch.serialization.add_safe_globals([
    'ultralytics.nn.tasks.SegmentationModel',
    'ultralytics.nn.tasks.DetectionModel',
    'ultralytics.nn.modules.head.Detect',
    'ultralytics.nn.modules.head.Segment',
    'ultralytics.nn.modules.conv.Conv',
    'ultralytics.nn.modules.block.C2f',
    'ultralytics.nn.modules.block.SPPF',
    'collections.OrderedDict',
])


class YOLOSegmentationDetector:
    """YOLOv8 Segmentation Model for Car Damage Detection"""
    
    def __init__(self, model_path: str, output_dir: str):
        """
        Initialize YOLO segmentation detector
        
        Args:
            model_path: Path to YOLOv8 segmentation model (.pt file)
            output_dir: Directory to save output images
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✅ YOLOv8 Segmentation model loaded from: {model_path}")
    
    def predict(self, image_array: np.ndarray) -> Dict:
        """
        Run segmentation inference on image
        
        Args:
            image_array: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary containing detection results
        """
        # Run inference
        results = self.model.predict(
            source=image_array,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        return self._extract_results(results, image_array)
    
    def _extract_results(self, results, original_image: np.ndarray) -> Dict:
        """Extract bounding boxes, masks, and metadata from YOLO results"""
        
        if len(results) == 0:
            return {
                'detections': [],
                'detection_count': 0,
                'has_masks': False,
                'annotated_image_path': None
            }
        
        result = results[0]
        detections = []
        masks_data = []
        
        # Extract bounding boxes
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x_min, y_min, x_max, y_max = map(int, box)
                
                detection = {
                    'id': i + 1,
                    'class_id': int(cls),
                    'class_name': self.model.names[int(cls)],
                    'confidence': float(conf),
                    'bbox': {
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'width': x_max - x_min,
                        'height': y_max - y_min,
                        'area': (x_max - x_min) * (y_max - y_min)
                    }
                }
                detections.append(detection)
        
        # Extract segmentation masks if available
        has_masks = False
        if result.masks is not None:
            has_masks = True
            masks = result.masks.data.cpu().numpy()
            
            for i, mask in enumerate(masks):
                # Resize mask to original image size
                mask_resized = cv2.resize(
                    mask, 
                    (original_image.shape[1], original_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Calculate mask area and contours
                mask_area = np.sum(mask_resized > 0.5)
                contours, _ = cv2.findContours(
                    (mask_resized > 0.5).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if i < len(detections):
                    detections[i]['mask'] = {
                        'area': int(mask_area),
                        'contour_count': len(contours),
                        'shape': mask_resized.shape
                    }
                
                masks_data.append(mask_resized)
        
        # Create annotated image
        annotated_path = self._create_annotated_image(
            original_image, 
            detections, 
            masks_data if has_masks else None,
            result
        )
        
        return {
            'detections': detections,
            'detection_count': len(detections),
            'has_masks': has_masks,
            'annotated_image_path': annotated_path,
            'model_info': {
                'conf_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold
            }
        }
    
    def _create_annotated_image(
        self, 
        image: np.ndarray, 
        detections: List[Dict],
        masks: List[np.ndarray] = None,
        result = None
    ) -> str:
        """Create and save annotated image with boxes and masks"""
        
        # Use YOLO's built-in plotting if available
        if result is not None and hasattr(result, 'plot'):
            annotated_img = result.plot(
                conf=True,
                labels=True,
                boxes=True,
                masks=True
            )
        else:
            # Manual annotation
            annotated_img = image.copy()
            
            # Draw masks first (as background)
            if masks is not None:
                for mask in masks:
                    # Create colored overlay
                    color_mask = np.zeros_like(annotated_img)
                    color_mask[mask > 0.5] = [0, 255, 0]  # Green mask
                    
                    # Blend with original image
                    annotated_img = cv2.addWeighted(
                        annotated_img, 0.7, color_mask, 0.3, 0
                    )
            
            # Draw bounding boxes and labels
            for detection in detections:
                bbox = detection['bbox']
                x_min = bbox['x_min']
                y_min = bbox['y_min']
                x_max = bbox['x_max']
                y_max = bbox['y_max']
                
                # Draw box
                cv2.rectangle(
                    annotated_img, 
                    (x_min, y_min), 
                    (x_max, y_max), 
                    (0, 255, 0), 
                    2
                )
                
                # Draw label
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Background for text
                cv2.rectangle(
                    annotated_img,
                    (x_min, y_min - text_h - 10),
                    (x_min + text_w, y_min),
                    (0, 255, 0),
                    -1
                )
                
                # Text
                cv2.putText(
                    annotated_img,
                    label,
                    (x_min, y_min - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness
                )
        
        # Save annotated image
        output_path = os.path.join(self.output_dir, "segmentation_result.jpg")
        cv2.imwrite(output_path, annotated_img)
        
        return "/output/segmentation/segmentation_result.jpg"


class SegmentationService:
    """Service wrapper for YOLOv8 segmentation inference"""
    
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            base_dir, 
            '../../../Middleware/models/stage3/best.pt'
        )
        self.output_dir = os.path.join(base_dir, '../../../output/segmentation')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize detector
        self.detector = YOLOSegmentationDetector(model_path, self.output_dir)
    
    def predict(self, img_bytes: bytes) -> Dict:
        """
        Run segmentation on image bytes
        
        Args:
            img_bytes: Image data as bytes
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Load and convert image
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_arr = np.array(img)
            
            # Convert RGB to BGR for OpenCV/YOLO
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            
            # Run detection
            result = self.detector.predict(img_arr)
            
            return {
                'success': True,
                'detections': result['detections'],
                'detection_count': result['detection_count'],
                'has_segmentation_masks': result['has_masks'],
                'annotated_image': result['annotated_image_path'],
                'model_info': result['model_info']
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'detection_count': 0
            }
    
    def set_thresholds(self, conf: float = None, iou: float = None):
        """Update confidence and IOU thresholds"""
        if conf is not None:
            self.detector.conf_threshold = conf
        if iou is not None:
            self.detector.iou_threshold = iou


