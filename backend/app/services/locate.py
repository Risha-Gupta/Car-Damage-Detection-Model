import os
import io
import uuid
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import torch
class Stage2DamageLocalizationDetector:
    def __init__(self, model_path: str, output_dir: str, device: str | int = "cpu"):
        # Resolve model path absolutely
        self.model_path = str(Path(model_path).resolve())
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Stage2 model not found at: {self.model_path}")

        # Load YOLO and set device
        self.detection_network = YOLO(self.model_path)
        self.device = device
        self.detection_confidence_threshold = 0.1

        # Output dir
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_damage_detection(self, image_array_bgr: np.ndarray) -> dict:
        """
        Feed original image to YOLO. Let it do its own resizing/letterboxing.
        Ultralytics returns boxes in original image coordinates.
        """
        if image_array_bgr is None or image_array_bgr.size == 0:
            raise ValueError("Empty image passed to detector")

        # YOLO accepts NumPy images. Use BGR as-is.
        results = self.detection_network.predict(
            source=image_array_bgr,
            conf=self.detection_confidence_threshold,
            verbose=False,
            device=self.device
        )

        return self._extract_detection_results(results)

    def _extract_detection_results(self, results) -> dict:
        detected_damage_bboxes = []

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return {
                "detected_damage_bboxes": [],
                "damage_regions_count": 0
            }

        r = results[0]
        # r.boxes.xyxy is Nx4 in original image space
        for i in range(len(r.boxes)):
            box = r.boxes[i]
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())

            x_min = int(max(0, np.floor(xyxy[0])))
            y_min = int(max(0, np.floor(xyxy[1])))
            x_max = int(max(x_min + 1, np.ceil(xyxy[2])))
            y_max = int(max(y_min + 1, np.ceil(xyxy[3])))

            detected_damage_bboxes.append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "confidence": round(conf, 4),
                "bbox_area": int((x_max - x_min) * (y_max - y_min))
            })

        return {
            "detected_damage_bboxes": detected_damage_bboxes,
            "damage_regions_count": len(detected_damage_bboxes),
        }

    def create_annotated_image(self, image_array_bgr: np.ndarray, detected_bboxes: list[str]) -> str:
        """
        Draw boxes and save. Returns an absolute file path you can map to a URL in your API layer.
        """
        annotated = image_array_bgr.copy()
        h, w = annotated.shape[:2]

        for idx, b in enumerate(detected_bboxes):
            x1, y1, x2, y2 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
            conf = b["confidence"]

            # Clamp to image bounds so cv2 doesn’t whine
            x1 = int(np.clip(x1, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y2 = int(np.clip(y2, 0, h - 1))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Damage #{idx+1}: {conf*100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thick = 1
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
            y_text_top = max(0, y1 - th - 6)

            cv2.rectangle(annotated, (x1, y_text_top), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 3), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

        fname = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
        save_path = self.output_dir / fname
        cv2.imwrite(str(save_path), annotated)

        return str(save_path)  # return absolute path; map to URL higher up


class LocationService:
    def __init__(self):
        here = Path(__file__).resolve()
        project_root = here.parents[3]  # Car-Damage-Detection-Model/

        model_path = project_root / "Middleware/models/stage2/models/model_weights.pt"
        output_dir = project_root / "output/stage2"

        # Pick device explicitly; change to 0 if you have a GPU
        device = "cpu"

        self.detector = Stage2DamageLocalizationDetector(
            model_path=str(model_path),
            output_dir=str(output_dir),
            device=device
        )
        self.output_dir = str(output_dir)

    def predict(self, img_bytes: bytes) -> dict:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)                    # RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            det = self.detector.run_damage_detection(img_bgr)
            annotated_path = self.detector.create_annotated_image(
                img_bgr, det["detected_damage_bboxes"]
            )

            # If your FastAPI serves /output as static, convert to URL here; otherwise return file path
            return {
                "detected_damage_bboxes": det["detected_damage_bboxes"],
                "damage_regions_count": det["damage_regions_count"],
                "annotated_image_path": annotated_path
            }
        except Exception as e:
            raise Exception(f"Stage 2 localization error: {e}")
