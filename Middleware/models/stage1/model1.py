import tensorflow as tf
import keras
from keras.applications import MobileNetV3Small
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2, numpy as np, os, json, keras

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
BASE_PATH = "Middleware/"
MODEL_NAME = "mobilenetv3_canny"

DAMAGED_DIR = os.path.join(BASE_PATH, "dataset/CarDD_COCO/train")
CLEAN_DIR   = os.path.join(BASE_PATH, "dataset/CarDD_COCO/clean_cars")

# -----------------------------
# LOAD IMAGE PATHS
# -----------------------------
def list_images(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

damaged_paths = list_images(DAMAGED_DIR)
clean_paths   = list_images(CLEAN_DIR)

print(f"✅ Damaged: {len(damaged_paths)}, Clean: {len(clean_paths)}")

# Oversample clean images (minority class)
oversample_factor = len(damaged_paths) // len(clean_paths)
clean_paths_oversampled = clean_paths * oversample_factor
# Add remaining samples
remaining = len(damaged_paths) - len(clean_paths_oversampled)
if remaining > 0:
    import random
    clean_paths_oversampled += random.sample(clean_paths, remaining)

print(f"   After oversampling: Clean images = {len(clean_paths_oversampled)}")

image_paths = np.array(damaged_paths + clean_paths_oversampled)
labels = np.array([1]*len(damaged_paths) + [0]*len(clean_paths_oversampled))

# Shuffle
indices = np.arange(len(labels))
np.random.shuffle(indices)
image_paths, labels = image_paths[indices], labels[indices]

# -----------------------------
# SPLIT TRAIN/VAL
# -----------------------------
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"📂 Train: {len(train_paths)} (Damaged: {sum(train_labels)}, Clean: {len(train_labels)-sum(train_labels)})")
print(f"   Val: {len(val_paths)} (Damaged: {sum(val_labels)}, Clean: {len(val_labels)-sum(val_labels)})")

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight = {i: class_weights[i] for i in range(len(class_weights))}
print(f"⚖️ Class weights: {class_weight}")

# -----------------------------
# DATA AUGMENTATION (for 3-channel RGB only)
# -----------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2)
])

# -----------------------------
# LOAD + PREPROCESS WITH CANNY
# -----------------------------
def preprocess_with_canny(img_path, label, augment=False):
    """Load image, apply Canny, create 6-channel input"""
    # Read and decode image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    
    # Apply augmentation BEFORE Canny (on RGB only)
    if augment:
        img = data_augmentation(img, training=True)
    
    # Convert to numpy for Canny
    img_np = tf.cast(img, tf.uint8).numpy()
    
    # Canny edge detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # (H, W, 3)
    edges_tensor = tf.convert_to_tensor(edges, dtype=tf.float32) / 255.0
    
    # Normalize RGB
    img_normalized = tf.cast(img, tf.float32) / 255.0
    
    # Concatenate RGB + edges → 6 channels
    combined = tf.concat([img_normalized, edges_tensor], axis=-1)
    
    return combined, label

def tf_preprocess_wrapper(path, label, augment):
    """Wrapper for tf.py_function with proper shape setting"""
    def _func(p, l, aug):
        result = preprocess_with_canny(p.numpy().decode('utf-8'), l.numpy(), aug.numpy())
        return result
    
    combined, label = tf.py_function(
        func=_func,
        inp=[path, label, augment],
        Tout=[tf.float32, tf.int32]
    ) # pyright: ignore[reportGeneralTypeIssues]
    
    combined.set_shape([*IMG_SIZE, 6])  
    label.set_shape([])
    
    return combined, label

# Create datasets
train_ds = (tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(2000)
    .map(lambda x, y: tf_preprocess_wrapper(x, y, tf.constant(True)), 
        num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(lambda x, y: tf_preprocess_wrapper(x, y, tf.constant(False)), 
         num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE))

# -----------------------------
# MODEL BUILD - CUSTOM INPUT FOR 6 CHANNELS
# -----------------------------
# Create custom input layer for 6 channels
input_layer = layers.Input(shape=(*IMG_SIZE, 6))

# Split into RGB and edges
rgb_channels = layers.Lambda(lambda x: x[:, :, :, :3])(input_layer)
edge_channels = layers.Lambda(lambda x: x[:, :, :, 3:])(input_layer)

# Process RGB through pretrained MobileNetV3
base_model = MobileNetV3Small(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

rgb_features = base_model(rgb_channels)
rgb_features = layers.GlobalAveragePooling2D()(rgb_features)

# Process edge channels separately
edge_conv = layers.Conv2D(32, 3, activation='relu', padding='same')(edge_channels)
edge_conv = layers.MaxPooling2D(2)(edge_conv)
edge_conv = layers.Conv2D(64, 3, activation='relu', padding='same')(edge_conv)
edge_conv = layers.GlobalAveragePooling2D()(edge_conv)

# Combine features
combined_features = layers.Concatenate()([rgb_features, edge_conv])

# Classification head
x = layers.Dense(256, activation="relu")(combined_features)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy", 
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name="auc")
    ]
)

print("\n📋 Model Summary:")
model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks_initial = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=5, mode="max", 
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(BASE_PATH, f"models/stage1/{MODEL_NAME}_best.keras"),
        monitor="val_auc", mode="max", save_best_only=True, verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir=os.path.join(BASE_PATH, 'logs'), histogram_freq=1
    )
]

print("\n🚀 Starting initial training with frozen base...")
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=INITIAL_EPOCHS, 
    class_weight=class_weight,  # Use class weights
    callbacks=callbacks_initial
)

print(f"\n✅ Initial training completed. Best val_auc: {max(history.history['val_auc']):.4f}")

save_path = os.path.join(BASE_PATH, f"models/stage1/{MODEL_NAME}_final.keras")
model.save(save_path)
print(f"\n✅ Final model saved to {save_path}")

# Compile final metrics
metrics = {
    "model": MODEL_NAME,
    "total_epochs": len(history.history["loss"]),
    "initial_epochs": INITIAL_EPOCHS,
    "finetune_epochs": len(history.history["loss"]) - INITIAL_EPOCHS,
    "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    "final_val_precision": float(history.history["val_precision"][-1]),
    "final_val_recall": float(history.history["val_recall"][-1]),
    "final_val_auc": float(history.history["val_auc"][-1]),
    "best_val_accuracy": float(max(history.history["val_accuracy"])),
    "best_val_auc": float(max(history.history["val_auc"])),
    "class_weights": class_weight,
    "dataset_info": {
        "train_damaged": int(sum(train_labels)),
        "train_clean": int(len(train_labels) - sum(train_labels)),
        "val_damaged": int(sum(val_labels)),
        "val_clean": int(len(val_labels) - sum(val_labels))
    }
}

metrics_path = os.path.join(BASE_PATH, f"models/stage1/metrics_{MODEL_NAME}.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n📊 Metrics saved to {metrics_path}")
