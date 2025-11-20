import numpy as np
from typing import Dict


# -------------------------------
# Stage 4: Damage Severity and Type Classification
# -------------------------------
class Stage4DamageClassifier:
    """
    Stage 4 classifier: Analyzes damage severity, type, coverage, and repair priority.
    
    Expects outputs from Stage 2 (localization/bboxes) and Stage 3 (segmentation) 
    as inputs, along with the original image for metrics calculation.
    """
    
    def __init__(self):
        self.severity_levels = {
            "minor": {"min_area": 0, "max_area": 5000},
            "moderate": {"min_area": 5000, "max_area": 20000},
            "severe": {"min_area": 20000, "max_area": float("inf")}
        }

    def classify_damage(
        self, 
        segmentation_data: dict, 
        bbox_data: dict, 
        image_array: np.ndarray
    ) -> dict:
        """
        Classify damage based on segmentation masks, bounding boxes, and image.
        
        Args:
            segmentation_data: Output from Stage 3 containing detections and mask info
            bbox_data: Output from Stage 2 containing bounding boxes and areas
            image_array: BGR image array for calculating coverage metrics
        
        Returns:
            dict with severity, type, coverage, priority, and detection count
        """
        detections = segmentation_data.get("detections", [])
        bboxes = bbox_data.get("detected_damage_bboxes", [])
        
        # Calculate total damage area and coverage percentage
        total_area = sum([b.get("bbox_area", 0) for b in bboxes])
        img_area = image_array.shape[0] * image_array.shape[1]
        coverage = (total_area / img_area * 100) if img_area > 0 else 0

        # Determine damage severity, type, and repair priority
        severity = self._assess_severity(total_area, len(detections))
        damage_type = self._estimate_damage_type(coverage)
        priority = self._calculate_priority(severity, coverage, len(detections))

        return {
            "damage_severity": severity,
            "damage_type": damage_type,
            "damage_coverage_percent": round(coverage, 2),
            "repair_priority": priority,
            "detection_count": len(detections)
        }

    def _assess_severity(self, total_area: float, detection_count: int) -> str:
        """
        Assess damage severity based on total area and detection count.
        
        Returns: "minor", "moderate", or "severe"
        """
        if total_area <= 5000 and detection_count <= 2:
            return "minor"
        elif total_area <= 20000 and detection_count <= 5:
            return "moderate"
        return "severe"

    def _estimate_damage_type(self, coverage: float) -> str:
        """
        Estimate damage type based on coverage percentage.
        
        Returns: "scratch", "crack", "dent", or "no_damage"
        """
        if coverage > 10:
            return "dent"
        elif coverage > 3:
            return "crack"
        elif coverage > 0:
            return "scratch"
        return "no_damage"

    def _calculate_priority(
        self, 
        severity: str, 
        coverage: float, 
        detection_count: int
    ) -> str:
        """
        Calculate repair priority based on severity, coverage, and detection count.
        
        Returns: "high", "medium", or "low"
        """
        if severity == "severe" or coverage > 15:
            return "high"
        elif severity == "moderate" or coverage > 5:
            return "medium"
        return "low"
