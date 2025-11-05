import os, io, cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from typing import Dict, List
import torch, ultralytics

class LocationService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "../../../Middleware/models/stage2/yolov8_damage_seg.pt")
        self.output_dir = os.path.join(base_dir, "../../../output/stage2")

        # Create output dirs
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "rois"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)

        # Fix PyTorch unsafe load issue
        torch.serialization.add_safe_globals([ultralytics.nn.tasks.SegmentationModel])

        self.model = YOLO(self.model_path)
        print(f"✅ Stage 2 model loaded: {self.model_path}")

    def _bytes_to_cv2(self, image_bytes: bytes):
        """Decode image bytes into cv2 BGR image"""
        img = Image.open(io.BytesIO(image_bytes).decode("RGB"))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def predict(self, image_bytes: bytes) -> Dict:
        img = self._bytes_to_cv2(image_bytes)
        h, w = img.shape[:2]

        result = self.model.predict(img, verbose=False)[0]

        annotated = img.copy()
        detections = []

        for i, (box, mask, conf) in enumerate(
            zip(result.boxes.xyxy, result.masks.data, result.boxes.conf)
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            conf = float(conf)

            # Resize mask
            mask_resized = cv2.resize(mask.cpu().numpy(), (w, h))
            mask_bin = (mask_resized > 0.5).astype(np.uint8)

            # Create colored mask overlay for annotated display
            annotated[mask_bin == 1] = (0, 0, 255)

            # Save ROI
            roi = img[y1:y2, x1:x2]
            roi_path = os.path.join(self.output_dir, "rois", f"roi_{i}.jpg")
            cv2.imwrite(roi_path, roi)

            # Save mask PNG
            mask_png_path = os.path.join(self.output_dir, "masks", f"mask_{i}.png")
            cv2.imwrite(mask_png_path, (mask_bin * 255))

            # Save mask NPY (for severity)
            mask_npy_path = os.path.join(self.output_dir, "masks", f"mask_{i}.npy")
            np.save(mask_npy_path, mask_bin)

            # Extract polygon (approx)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = contours[0].reshape(-1, 2).tolist() if len(contours) else []

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "polygon": polygon,
                "roi_path": roi_path,
                "mask_png": mask_png_path,
                "mask_npy": mask_npy_path
            })

        # Save annotated image
        annotated_path = os.path.join(self.output_dir, "annotated.jpg")
        cv2.imwrite(annotated_path, annotated)

        return {
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image": annotated_path
        }
