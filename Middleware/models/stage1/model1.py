import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV3Small
from tensorflow.keras import layers, models, optimizers
import os, json, random


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_NAME = "resnet50"   
base_path = "Middleware/"
DAMAGED_DIR = base_path + "dataset/CarDD_COCO/train"   
CLEAN_DIR   = base_path + "dataset/CarDD_COCO/clean_cars"  


damaged_paths = [os.path.join(DAMAGED_DIR, f) for f in os.listdir(DAMAGED_DIR)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
clean_paths   = [os.path.join(CLEAN_DIR, f) for f in os.listdir(CLEAN_DIR)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

image_paths = damaged_paths + clean_paths
labels = [1] * len(damaged_paths) + [0] * len(clean_paths)

print(f"✅ Damaged images: {len(damaged_paths)}, Clean images: {len(clean_paths)}")

# -----------------------------
# STEP 2 — Manual Train/Val Split (80/20)
# -----------------------------
data = list(zip(image_paths, labels))
random.shuffle(data)

split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

train_paths, train_labels = zip(*train_data)
val_paths, val_labels = zip(*val_data)

print(f"📂 Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")

def load_and_preprocess(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((list(train_paths), list(train_labels)))
val_ds   = tf.data.Dataset.from_tensor_slices((list(val_paths), list(val_labels)))

train_ds = (train_ds
            .shuffle(1000)
            .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (val_ds
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))

base_model = ResNet50(
        weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

save_path = base_path + "/models/stage1/" + f"{MODEL_NAME}_damage.keras"
model.save(save_path)
print(f"✅ Model saved to {save_path}")

metrics = {
    "final_accuracy": float(history.history['val_accuracy'][-1]),
    "final_precision": float(history.history['val_precision'][-1]),
    "final_recall": float(history.history['val_recall'][-1])
}
with open(base_path+"/models/stage1/"+"metrics_stage1.json", "a") as f:
    json.dump(metrics, f, indent=2)
print("📊 Metrics saved to metrics_stage1.json")
