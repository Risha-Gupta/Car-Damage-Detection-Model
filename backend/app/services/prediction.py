import numpy as np
from PIL import Image
import io, os, cv2
import keras
from typing import Dict


class PredictionService:
    """Stage 1: Binary damage classification (is_damaged or not_damaged)."""
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(
            base_dir,
            "../../../Middleware/models/stage1/models/mobilenetv3_canny_final.keras"
        )
        self.threshold = 0.82
        self.img_size = 224
        self.model = self._load_model()

        print(f"Stage 1 Model loaded successfully from: {self.model_path}")

    def _load_model(self):
        try:
            return keras.models.load_model(self.model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {self.model_path}: {e}"
            )

    def preprocess_image(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((self.img_size, self.img_size))

        img_rgb = np.array(image, dtype=np.uint8)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 120)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        img_rgb = img_rgb.astype(np.float32) / 255.0
        edges = edges.astype(np.float32) / 255.0

        img_rgb = np.expand_dims(img_rgb, axis=0)
        edges = np.expand_dims(edges, axis=0)

        return img_rgb, edges
    def save_preprocessed(self, img_rgb: np.ndarray, edges: np.ndarray, filename: str):
        # Convert arrays from batch (1, h, w, 3) to (h, w, 3)
        rgb_img = (img_rgb[0] * 255).astype(np.uint8)
        edges_img = (edges[0] * 255).astype(np.uint8)

        # Path to save folder
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed")
        os.makedirs(save_dir, exist_ok=True)

        rgb_path = os.path.join(save_dir, f"rgb_{filename}.png")
        edges_path = os.path.join(save_dir, f"edges_{filename}.png")

        # Save using cv2
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(edges_path, cv2.cvtColor(edges_img, cv2.COLOR_RGB2BGR))

        return rgb_path, edges_path

    def predict(self, image_bytes: bytes) -> Dict:
        img_rgb, edges = self.preprocess_image(image_bytes)
        filename = f"img_{np.random.randint(1000000)}"

        # Save preprocessed outputs
        rgb_path, edges_path = self.save_preprocessed(img_rgb, edges, filename)
        prob = float(self.model.predict([img_rgb, edges], verbose=0)[0][0]) # type: ignore
        is_damaged = prob > self.threshold

        return {
            "is_damaged": bool(is_damaged),
            "confidence": round(prob, 4),
            "damage_probability": round(prob, 4),
            "status": "damaged" if is_damaged else "not_damaged"
        }