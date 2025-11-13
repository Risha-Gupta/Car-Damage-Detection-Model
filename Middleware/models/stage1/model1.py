import tensorflow as tf
import keras
from keras.applications import MobileNetV3Small
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import cv2, numpy as np, os, random

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FREEZE = 3           
EPOCHS_FINETUNE = 10
BASE_PATH = "../Middleware/"
MODEL_NAME = "mobilenetv3_canny_final"

DAMAGED_DIR = os.path.join(BASE_PATH, "dataset/CarDD_COCO/train")
CLEAN_DIR   = os.path.join(BASE_PATH, "dataset/CarDD_COCO/clean_cars")

os.makedirs(os.path.join(BASE_PATH, "models/stage1"), exist_ok=True)

def list_images(directory):
    if not os.path.exists(directory):
        print(f"⚠️ Missing directory: {directory}")
        return []
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

damaged_paths = list_images(DAMAGED_DIR)
clean_paths   = list_images(CLEAN_DIR)
print(f"Images Loaded -> Damaged: {len(damaged_paths)}, Clean: {len(clean_paths)}")

# Balance dataset by trimming
min_len = min(len(damaged_paths), len(clean_paths))
damaged_paths = random.sample(damaged_paths, min_len)
clean_paths   = random.sample(clean_paths, min_len)

image_paths = np.array(damaged_paths + clean_paths)
labels = np.array([1]*min_len + [0]*min_len)

# Shuffle & split
indices = np.arange(len(labels))
np.random.shuffle(indices)
image_paths, labels = image_paths[indices], labels[indices]

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

# -----------------------------
# DATA PIPELINE
# -----------------------------
def preprocess_tf(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE)

    # Small augmentations
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)

    return img

def extract_canny(img):
    img_np = tf.cast(img, tf.uint8).numpy()
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Softer Canny
    edges = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(edges, 50, 120)

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

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(2000)
    .map(wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# -----------------------------
# MODEL
# -----------------------------
input_rgb = layers.Input(shape=(*IMG_SIZE, 3), name="rgb_input")
input_edges = layers.Input(shape=(*IMG_SIZE, 3), name="edges_input")

# RGB -> MobileNetV3
base_model = MobileNetV3Small(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

base_model.trainable = False
rgb_features = base_model(input_rgb)
rgb_features = layers.GlobalAveragePooling2D()(rgb_features)
rgb_features = layers.LayerNormalization()(rgb_features)

# Edge CNN
edge_branch = layers.Conv2D(32, 3, padding='same')(input_edges)
edge_branch = layers.LeakyReLU()(edge_branch)
edge_branch = layers.MaxPooling2D(2)(edge_branch)
edge_branch = layers.Conv2D(64, 3, padding='same')(edge_branch)
edge_branch = layers.LeakyReLU()(edge_branch)
edge_branch = layers.GlobalAveragePooling2D()(edge_branch)
edge_branch = layers.LayerNormalization()(edge_branch)

# Combine
combined = layers.Concatenate()([rgb_features, edge_branch])
x = layers.Dense(256)(combined)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128)(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.3)(x)

output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=[input_rgb, input_edges], outputs=output)

model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(BASE_PATH, f"models/stage1/{MODEL_NAME}_best.keras"),
        monitor="val_auc", mode="max", save_best_only=True
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=5, mode="max", restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
    )
]

print("\n🚀 Stage 1: Frozen base training...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FREEZE, callbacks=callbacks)

# Unfreeze and finetune
base_model.trainable = True
for layer in base_model.layers[:60]:  # freeze only low-level layers
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

print("\n🎯 Stage 2: Fine-tuning...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, callbacks=callbacks)

# Save final model
save_path = os.path.join(BASE_PATH, f"models/stage1/{MODEL_NAME}.keras")
model.save(save_path)
print(f"\n✅ Saved final model to {save_path}")
