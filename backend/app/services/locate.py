import os
import io
import uuid
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import sys

stage2_dir = Path(__file__).resolve().parents[3] / "Middleware/models/stage2"
sys.path.insert(0, str(stage2_dir))
try:
    from ultralytics.nn.tasks import SegmentationModel, DetectionModel
    torch.serialization.add_safe_globals([SegmentationModel, DetectionModel])
except Exception:
    pass  
from models.detectron2_detector import DamageLocalizationModel


class Stage2DamageLocalizationDetector:
    def __init__(self, model_path: str, output_dir: str, device: str = "cpu"):
        # Resolve model path absolutely
        self.model_path = str(Path(model_path).resolve())
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Stage2 model not found at: {self.model_path}")

        self.detection_network = DamageLocalizationModel(
            model_version="ResNet50",
            use_pretrained_weights=False
        )
        self.detection_network.load_model_checkpoint(self.model_path)
        self.detection_network.model.eval()
        
        self.device = device
        self.detection_confidence_threshold = 0.5

        # Output dir
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_damage_detection(self, image_array_bgr: np.ndarray) -> dict:
        """
        Run Detectron2 inference on the image.
        """
        if image_array_bgr is None or image_array_bgr.size == 0:
            raise ValueError("Empty image passed to detector")

        # Convert BGR to RGB for Detectron2
        image_rgb = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2RGB)
        
        # Prepare input for Detectron2
        height, width = image_rgb.shape[:2]
        
        # Create input dict for Detectron2
        inputs = {
            "image": torch.as_tensor(image_rgb.transpose(2, 0, 1).astype("float32")),
            "height": height,
            "width": width
        }
        
        # Run inference
        with torch.no_grad():
            predictions = self.detection_network.model([inputs])[0]

        return self._extract_detection_results(predictions, image_array_bgr.shape)

    def _extract_detection_results(self, predictions, image_shape) -> dict:
        detected_damage_bboxes = []
        
        if "instances" not in predictions:
            return {
                "detected_damage_bboxes": [],
                "damage_regions_count": 0
            }
        
        instances = predictions["instances"]
        
        if len(instances) == 0:
            return {
                "detected_damage_bboxes": [],
                "damage_regions_count": 0
            }
        
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        
        h, w = image_shape[:2]
        
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if score < self.detection_confidence_threshold:
                continue
                
            x_min = int(max(0, np.floor(box[0])))
            y_min = int(max(0, np.floor(box[1])))
            x_max = int(min(w, np.ceil(box[2])))
            y_max = int(min(h, np.ceil(box[3])))

            detected_damage_bboxes.append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "confidence": round(float(score), 4),
                "bbox_area": int((x_max - x_min) * (y_max - y_min)),
                "class_id": int(cls)
            })

        return {
            "detected_damage_bboxes": detected_damage_bboxes,
            "damage_regions_count": len(detected_damage_bboxes),
        }

    def create_annotated_image(self, image_array_bgr: np.ndarray, detected_bboxes: list) -> str:
        """
        Draw boxes and save. Returns an absolute file path you can map to a URL in your API layer.
        """
        annotated = image_array_bgr.copy()
        h, w = annotated.shape[:2]

        for idx, b in enumerate(detected_bboxes):
            x1, y1, x2, y2 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
            conf = b["confidence"]
            class_id = b.get("class_id", 0)
            x1 = int(np.clip(x1, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y2 = int(np.clip(y2, 0, h - 1))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{conf*100:.1f}%"
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

        return f"/output/stage2/{fname}"


class LocationService:
    def __init__(self):
        here = Path(__file__).resolve()
        project_root = here.parents[3]  # Car-Damage-Detection-Model/

        model_path = project_root / "Middleware/models/stage2/models/model_final.pth"
        output_dir = project_root / "output/stage2"
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
            img_np = np.array(img)  # RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            det = self.detector.run_damage_detection(img_bgr)
            annotated_path = self.detector.create_annotated_image(
                img_bgr, det["detected_damage_bboxes"]
            )

            return {
                "detected_damage_bboxes": det["detected_damage_bboxes"],
                "damage_regions_count": det["damage_regions_count"],
                "annotated_image_path": annotated_path
            }
        except Exception as e:
            raise Exception(f"Stage 2 localization error: {e}")