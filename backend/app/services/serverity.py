import os
import cv2
import base64
import numpy as np
from datetime import datetime
from ultralytics import YOLO

class Stage4DamageClassifier:
    """
    Stage 4 final version:
    - Takes segmentation output from Stage 3
    - Loads YOLO (car part detection) internally
    - Adds damaged car part info on top of Stage 3 detections
    - Adds severity, priority, coverage, ROI images
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # YOLO part detection model path
        self.model_path = os.path.join(base_dir, "../../../Middleware/models/stage4/best.pt")
        self.model = None
        self._loaded = False

        # ROI output folder
        self.output_dir = os.path.join(base_dir, "../../../outputs/stage4_rois")
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        if not self._loaded:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"YOLO model not found at {self.model_path}")
            self.model = YOLO(self.model_path)
            self._loaded = True

    def _run_part_detection(self, image_array):
        """Runs YOLO to detect car parts."""
        self._load_model()

        results = self.model.predict(image_array, save=False)[0]
        class_names = results.names

        bboxes = np.array(results.boxes.xyxy.cpu()).astype(int)
        class_ids = np.array(results.boxes.cls.cpu()).astype(int)
        scores = np.array(results.boxes.conf.cpu()).astype(float)

        parts = []
        for i, box in enumerate(bboxes):
            parts.append({
                "bbox": box.tolist(),
                "class_id": int(class_ids[i]),
                "class_name": class_names[int(class_ids[i])],
                "confidence": float(scores[i])
            })

        return parts

    def classify_damage(self, segmentation_data, bbox_data, image_array):
        """
        segmentation_data: output from Stage 3
        bbox_data: Stage 3 bounding boxes
        image_array: original image (BGR)
        """

        # --- 1. YOLO part detection ---
        part_detections = self._run_part_detection(image_array)

        stage3_bboxes = bbox_data.get("detected_damage_bboxes", [])
        total_area = sum(b["bbox_area"] for b in stage3_bboxes)
        img_area = image_array.shape[0] * image_array.shape[1]
        coverage = (total_area / img_area * 100) if img_area else 0

        # --- 2. severity ---
        severity = (
            "minor" if total_area <= 5000 else
            "moderate" if total_area <= 20000 else
            "severe"
        )

        # --- 3. type ---
        if coverage > 10:
            damage_type = "dent"
        elif coverage > 3:
            damage_type = "crack"
        elif coverage > 0:
            damage_type = "scratch"
        else:
            damage_type = "no_damage"

        # --- 4. priority ---
        if severity == "severe" or coverage > 15:
            priority = "high"
        elif severity == "moderate" or coverage > 5:
            priority = "medium"
        else:
            priority = "low"

        # --- 5. Merge Stage 3 damage with YOLO part detection ---
        merged = []
        for idx, b in enumerate(stage3_bboxes):
            best_part = None
            bx1, by1, bx2, by2 = b["bbox"]

            for part in part_detections:
                px1, py1, px2, py2 = part["bbox"]

                # IOU check (lazy but works)
                if not (px2 < bx1 or px1 > bx2 or py2 < by1 or py1 > by2):
                    best_part = part
                    break

            merged.append({
                "damage_bbox": b["bbox"],
                "damage_area": b["bbox_area"],
                "part_detected": best_part
            })

        # --- 6. ROI extraction ---
        roi_images = self._extract_rois(image_array, merged)

        return {
            "severity": severity,
            "damage_type": damage_type,
            "coverage_percent": round(coverage, 2),
            "repair_priority": priority,
            "merged_results": merged,
            "roi_images": roi_images,
            "car_part_detection_count": len(part_detections)
        }

    def _extract_rois(self, image, merged):
        roi_list = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, m in enumerate(merged):
            x1, y1, x2, y2 = m["damage_bbox"]
            roi = image[y1:y2, x1:x2].copy()

            filename = f"roi_{ts}_{i+1}.jpg"
            path = os.path.join(self.output_dir, filename)
            cv2.imwrite(path, roi)

            _, buff = cv2.imencode(".jpg", roi)
            b64 = base64.b64encode(buff).decode()

            roi_list.append({
                "roi_id": i+1,
                "damage_bbox": m["damage_bbox"],
                "part_detected": m["part_detected"],
                "base64": b64,
                "file_path": path
            })

        return roi_list
