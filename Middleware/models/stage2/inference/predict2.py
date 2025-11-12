import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

# ==========================================================
# Dynamically add correct paths so imports work
# ==========================================================
CURRENT_FILE = Path(__file__).resolve()
STAGE2_DIR = CURRENT_FILE.parents[1]          # .../stage2
MIDDLEWARE_DIR = STAGE2_DIR.parents[1]        # .../models
PROJECT_ROOT = MIDDLEWARE_DIR.parent          # .../Middleware

# Add key paths to sys.path if not already present
for path in [STAGE2_DIR, MIDDLEWARE_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ==========================================================
# Imports using your real folder structure
# ==========================================================
from models.detectron2_detector import DamageLocalizationModel
from data_pipeline.preprocessor import ImagePreprocessor
from project_logging.performance_logger import PerformanceLogger


# ==========================================================
# Stage 2 Damage Localization Pipeline
# ==========================================================
class Stage2DamageLocalizationPipeline:
    """
    Handles YOLOv8-based damage localization only (Stage 2).
    """

    def __init__(self, model_checkpoint_path: str):
        model_checkpoint = Path(model_checkpoint_path)
        if not model_checkpoint.exists():
            raise FileNotFoundError(f"Stage 2 model checkpoint not found: {model_checkpoint}")

        # Initialize YOLOv8 model
        self.localization_network = DamageLocalizationModel(model_version="m", use_pretrained_weights=True)
        self.localization_network.load_model_checkpoint(str(model_checkpoint))

        # Supporting utilities
        self.image_processor = ImagePreprocessor(target_size=(512, 512))
        self.performance_logger = PerformanceLogger()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict_damage(self, car_image: np.ndarray, conf_threshold: float = 0.5) -> Dict:
        """Run damage detection on a single car image."""
        total_start = time.time()

        # --- Preprocessing ---
        t0 = time.time()
        processed_img = self.image_processor.process(car_image)
        img_tensor = (
            torch.from_numpy(processed_img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        preprocess_time = time.time() - t0

        # --- Inference ---
        t1 = time.time()
        with torch.no_grad():
            predictions = self.localization_network.get_model_instance()(img_tensor)
        inference_time = time.time() - t1

        # --- Postprocessing ---
        t2 = time.time()
        detected_bboxes = []
        detected_scores = []

        if hasattr(predictions, "boxes"):
            for box in predictions.boxes:
                conf = float(box.conf[0])
                if conf >= conf_threshold:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    detected_bboxes.append(xyxy)
                    detected_scores.append(conf)

        postprocess_time = time.time() - t2
        total_time = time.time() - total_start

        return {
            "num_detections": len(detected_bboxes),
            "bboxes": detected_bboxes,
            "confidence_scores": detected_scores,
            "execution_time": round(total_time, 4),
            "timing_breakdown": {
                "preprocessing": round(preprocess_time, 4),
                "inference": round(inference_time, 4),
                "postprocessing": round(postprocess_time, 4),
            },
        }

    def predict_batch(self, image_list: List[np.ndarray], conf_threshold: float = 0.5) -> List[Dict]:
        """Run localization on multiple images."""
        return [self.predict_damage(img, conf_threshold) for img in image_list]



