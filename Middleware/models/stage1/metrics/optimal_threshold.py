import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import roc_curve
import os

IMG_SIZE = (224, 224)
MODEL_PATH = "../Middleware/models/stage1/mobilenetv3_canny_final.keras"

# -----------------------------
# Load model
# -----------------------------
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# -----------------------------
# Load validation dataset
# This must match your training pipeline.
# -----------------------------
def preprocess_tf(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img

def extract_canny(img):
    img_np = tf.cast(img, tf.uint8).numpy()
    import cv2
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 120)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges.astype(np.float32) / 255.0

def wrapper(path, label):
    img = preprocess_tf(path)
    edges = tf.py_function(
        func=lambda x: extract_canny(x),
        inp=[img],
        Tout=tf.float32
    )
    img = tf.cast(img, tf.float32) / 255.0
    edges.set_shape([*IMG_SIZE, 3])
    return (img, edges), label

BASE_PATH = "../Middleware/"
DAMAGED_DIR = os.path.join(BASE_PATH, "dataset/CarDD_COCO/train")
CLEAN_DIR   = os.path.join(BASE_PATH, "dataset/CarDD_COCO/clean_cars")

def list_images(directory):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

damaged = list_images(os.path.join(BASE_PATH, "dataset/CarDD_COCO/train"))
clean   = list_images(os.path.join(BASE_PATH, "dataset/CarDD_COCO/clean_cars"))

# Assume you already matched counts.
paths = np.array(damaged + clean)
labels = np.array([1]*len(damaged) + [0]*len(clean))

from sklearn.model_selection import train_test_split
_, val_paths, _, val_labels = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# -----------------------------
# Compute best threshold
# -----------------------------
print("Computing optimal threshold...")

y_true = []
y_pred = []

for (rgb, edges), lbl in val_ds:
    probs = model.predict([rgb, edges], verbose=0).ravel()
    y_true.extend(lbl.numpy())
    y_pred.extend(probs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
j_scores = tpr - fpr

best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print("\n=============================")
print(" Optimal Threshold Found:")
print("  -->", float(best_threshold))
print("=============================\n")
