import numpy as np
import os
import csv
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
import keras

MODEL_PATH = "../Middleware/models/stage1/models/mobilenetv3_canny_final.keras"
THRESHOLD = 0.82  
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded.")

IMG_SIZE = (224, 224)
import cv2
from PIL import Image

def preprocess_image_path(path):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_rgb = np.array(image, dtype=np.uint8)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 120)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    edges = edges.astype(np.float32) / 255.0
    return np.expand_dims(img_rgb, 0), np.expand_dims(edges, 0)

BASE_PATH = "../Middleware/"
def list_images(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

damaged = list_images(os.path.join(BASE_PATH, "dataset/CarDD_COCO/train"))
clean   = list_images(os.path.join(BASE_PATH, "dataset/CarDD_COCO/clean_cars"))

paths = np.array(damaged + clean)
labels = np.array([1]*len(damaged) + [0]*len(clean))

from sklearn.model_selection import train_test_split
_, val_paths, _, val_labels = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42
)
y_true = []
y_prob = []
rows = []
for p, lbl in zip(val_paths, val_labels):
    rgb, ed = preprocess_image_path(p)
    prob = float(model.predict([rgb, ed], verbose=0)[0][0])
    y_true.append(lbl)
    y_prob.append(prob)
    rows.append((p, lbl, prob))

y_true = np.array(y_true)
y_prob = np.array(y_prob)
y_pred = (y_prob >= THRESHOLD).astype(int)

cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
acc = (y_true == y_pred).mean()

print("Confusion matrix:\n", cm)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")

out_csv = "val_predictions_threshold.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label", "prob"])
    writer.writerows(rows)

print("Saved predictions to", out_csv)
