import os
import json
import shutil
from tqdm import tqdm
from PIL import Image

# === SOURCE DATA ===
current_working_directory = os.getcwd()

# Print the current working directory
print("Current working directory:", current_working_directory)
BASE = "dataset/CarDD_COCO/"

TRAIN_JSON = os.path.join(BASE,"annotations/instances_train2017.json")
VAL_JSON = os.path.join(BASE,"annotations/instances_val2017.json")
TRAIN_IMG_DIR = os.path.join(BASE,"train")
VAL_IMG_DIR = os.path.join(BASE,"val")

# === OUTPUT YOLO DATASET ===
OUT_DIR = "datasets/damage_seg"
OUT = {
    "train": {
        "img": os.path.join(OUT_DIR, "images/train"),
        "lbl": os.path.join(OUT_DIR, "labels/train")
    },
    "val": {
        "img": os.path.join(OUT_DIR, "images/val"),
        "lbl": os.path.join(OUT_DIR, "labels/val")
    }
}

# Create dirs
for split in OUT.values():
    os.makedirs(split["img"], exist_ok=True)
    os.makedirs(split["lbl"], exist_ok=True)

def process_split(json_path, img_src, split):
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    for ann in tqdm(annotations, desc=f"Converting {split}"):
        img_info = images[ann["image_id"]]
        filename = img_info["file_name"]
        img_path = os.path.join(img_src, filename)

        # Copy image
        out_img = os.path.join(OUT[split]["img"], filename)
        if not os.path.exists(out_img):
            shutil.copyfile(img_path, out_img)

        # Load image to normalize coords
        with Image.open(img_path) as img:
            w, h = img.size

        # Normalized YOLO bbox
        x, y, bw, bh = ann["bbox"]
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        bw /= w
        bh /= h

        # Normalized polygon
        seg = ann["segmentation"][0]
        seg_norm = [seg[i] / (w if i % 2 == 0 else h) for i in range(len(seg))]

        # Save label file
        lbl_path = os.path.join(OUT[split]["lbl"], filename.replace(".jpg", ".txt"))
        with open(lbl_path, "a") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} ")
            f.write(" ".join([f"{v:.6f}" for v in seg_norm]) + "\n")

process_split(TRAIN_JSON, TRAIN_IMG_DIR, "train")
process_split(VAL_JSON, VAL_IMG_DIR, "val")

print("✅ Conversion complete!")
print(f"Training images: {OUT['train']['img']}")
print(f"Validation images: {OUT['val']['img']}")
