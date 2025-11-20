import numpy as np
import cv2
import os
import base64
from typing import Dict, List, Tuple
from datetime import datetime


class Stage4DamageClassifier:
    """Stage 4: Classify damage severity and type."""
    def __init__(self):
        self.severity_levels = {
            "minor": {"min_area": 0, "max_area": 5000},
            "moderate": {"min_area": 5000, "max_area": 20000},
            "severe": {"min_area": 20000, "max_area": float("inf")}
        }
        
        # Create output directory for ROI images
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, "../../../outputs/stage4_rois")
        os.makedirs(self.output_dir, exist_ok=True)

    def classify_damage(self, segmentation_data: dict, bbox_data: dict, image_array: np.ndarray) -> dict:
        """
        Classify damage severity and type based on segmentation and bounding box data.
        
        Args:
            segmentation_data: Dictionary containing detection results
            bbox_data: Dictionary containing bounding box information
            image_array: Original image as numpy array
            
        Returns:
            Dictionary with damage classification results and ROI images
        """
        detections = segmentation_data.get("detections", [])
        bboxes = bbox_data.get("detected_damage_bboxes", [])
        total_area = sum([b.get("bbox_area", 0) for b in bboxes])
        img_area = image_array.shape[0] * image_array.shape[1]
        coverage = (total_area / img_area * 100) if img_area > 0 else 0

        severity = self._assess_severity(total_area, len(detections))
        damage_type = self._estimate_damage_type(coverage)
        priority = self._calculate_priority(severity, coverage, len(detections))
        
        # Extract ROI images
        roi_images = self._extract_roi_images(image_array, bboxes, detections)

        return {
            "damage_severity": severity,
            "damage_type": damage_type,
            "damage_coverage_percent": round(coverage, 2),
            "repair_priority": priority,
            "detection_count": len(detections),
            "roi_images": roi_images,
            "roi_count": len(roi_images)
        }

    def _assess_severity(self, total_area: float, count: int) -> str:
        """Assess damage severity based on total area and detection count."""
        if total_area <= 5000 and count <= 2:
            return "minor"
        elif total_area <= 20000 and count <= 5:
            return "moderate"
        return "severe"

    def _estimate_damage_type(self, coverage: float) -> str:
        """Estimate damage type based on coverage percentage."""
        if coverage > 10:
            return "dent"
        elif coverage > 3:
            return "crack"
        elif coverage > 0:
            return "scratch"
        return "no_damage"

    def _calculate_priority(self, severity: str, coverage: float, count: int) -> str:
        """Calculate repair priority based on severity, coverage, and detection count."""
        if severity == "severe" or coverage > 15:
            return "high"
        elif severity == "moderate" or coverage > 5:
            return "medium"
        return "low"
    
    def _extract_roi_images(self, image_array: np.ndarray, bboxes: List[dict], detections: List[dict]) -> List[dict]:
        """
        Extract and save zoomed ROI images for each detected damage region.
        
        Args:
            image_array: Original image as numpy array (BGR format)
            bboxes: List of bounding boxes with areas
            detections: List of detection results
            
        Returns:
            List of dictionaries containing ROI information
        """
        roi_images = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, bbox_data in enumerate(bboxes):
            try:
                bbox = bbox_data.get("bbox", [])
                if len(bbox) < 4:
                    continue
                
                # Extract coordinates
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                # Add padding around ROI (15% on each side)
                h, w = image_array.shape[:2]
                padding_x = int((x2 - x1) * 0.15)
                padding_y = int((y2 - y1) * 0.15)
                
                x1_pad = max(0, x1 - padding_x)
                y1_pad = max(0, y1 - padding_y)
                x2_pad = min(w, x2 + padding_x)
                y2_pad = min(h, y2 + padding_y)
                
                # Extract ROI
                roi = image_array[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                
                if roi.size == 0:
                    continue
                
                # Draw bounding box on ROI (adjust coordinates relative to ROI)
                bbox_x1 = x1 - x1_pad
                bbox_y1 = y1 - y1_pad
                bbox_x2 = x2 - x1_pad
                bbox_y2 = y2 - y1_pad
                
                cv2.rectangle(roi, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)
                
                # Add label
                detection_info = detections[idx] if idx < len(detections) else {}
                confidence = detection_info.get("confidence", 0)
                class_name = detection_info.get("class_name", "damage")
                
                label = f"{class_name}: {confidence*100:.1f}%"
                cv2.putText(roi, label, (bbox_x1, bbox_y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Save ROI image
                filename = f"roi_{timestamp}_{idx+1}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, roi)
                
                # Convert to base64 for API response (optional)
                _, buffer = cv2.imencode('.jpg', roi)
                roi_base64 = base64.b64encode(buffer).decode('utf-8')
                
                roi_info = {
                    "roi_id": idx + 1,
                    "bbox": bbox,
                    "bbox_area": bbox_data.get("bbox_area", 0),
                    "roi_dimensions": {
                        "width": x2_pad - x1_pad,
                        "height": y2_pad - y1_pad
                    },
                    "file_path": filepath,
                    "relative_path": f"/outputs/stage4_rois/{filename}",
                    "base64": roi_base64,  # Include base64 for direct display
                    "confidence": round(confidence, 4) if confidence else None,
                    "class_name": class_name
                }
                
                roi_images.append(roi_info)
                
            except Exception as e:
                print(f"Error extracting ROI {idx}: {str(e)}")
                continue
        
        return roi_images