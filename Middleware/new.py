import os
import tensorflow as tf
from tqdm import tqdm

tfrecord_dir = "Middleware/cars196/2.0.0"
output_dir = "Middleware/cars196_extracted"
os.makedirs(output_dir, exist_ok=True)
import tensorflow as tf

sample = next(tf.data.TFRecordDataset("Middleware/cars196/2.0.0/cars196-train.tfrecord-00000-of-00008").as_numpy_iterator())
ex = tf.train.Example.FromString(sample)

print("Available keys:\n", ex.features.feature.keys())

# get all TFRecord files
tfrecords = [os.path.join(tfrecord_dir, f) 
            for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecord")]

feature_desc = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}

def parse(example):
    return tf.io.parse_single_example(example, feature_desc)

i = 0
for tfrecord in tfrecords:
    for record in tf.data.TFRecordDataset(tfrecord):
        ex = parse(record)
        img_bytes = ex["image"].numpy()
        out_path = os.path.join(output_dir, f"img_{i}.jpg")

        with open(out_path, "wb") as f:
            f.write(img_bytes)

        i += 1

print("Done. Extracted:", i, "images")
