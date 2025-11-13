import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

IMG_SIZE = (224, 224)
THRESHOLD = 0.82

MODEL_PATH = "../Middleware/models/stage1/mobilenetv3_canny_final.keras"

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded.")

def preprocess_image(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_rgb = np.array(image, dtype=np.uint8)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 120)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    edges = edges.astype(np.float32) / 255.0
    return np.expand_dims(img_rgb, axis=0), np.expand_dims(edges, axis=0)

def predict_single(image_path):
    rgb, ed = preprocess_image(image_path)
    prob = float(model.predict([rgb, ed], verbose=0)[0][0])
    is_damaged = prob > THRESHOLD

    return {
        "file": image_path,
        "probability": round(prob, 4),
        "label": "damaged" if is_damaged else "clean"
    }


def predict_folder(folder_path):
    results = []
    for f in os.listdir(folder_path):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder_path, f)
            results.append(predict_single(path))
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()

    if args.image:
        out = predict_single(args.image)
        print(out)

    elif args.folder:
        outs = predict_folder(args.folder)
        for item in outs:
            print(item)

    else:
        print("Provide --image or --folder")
