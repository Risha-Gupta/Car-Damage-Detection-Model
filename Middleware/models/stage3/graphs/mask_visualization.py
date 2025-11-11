import os
import cv2
import numpy as np

# === PATHS ===
IMG_PATH = "datasets/damage_seg/images/train"   
LBL_PATH = "datasets/damage_seg/labels/train"

# === CONFIG ===
IMAGE_NAME = "000001.jpg"  
SHOW = True                
def load_yolo_label(label_path, img_w, img_h):
    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            cls, xc, yc, w, h = parts[:5]
            poly = parts[5:]

            # bounding box in pixel coords
            bw, bh = w * img_w, h * img_h
            bx, by = (xc * img_w) - bw / 2, (yc * img_h) - bh / 2
            bbox = [int(bx), int(by), int(bx + bw), int(by + bh)]

            # polygon segmentation
            poly_px = [
                (int(poly[i] * img_w), int(poly[i + 1] * img_h))
                for i in range(0, len(poly), 2)
            ]

            boxes.append((bbox, poly_px))
    return boxes


def visualize():
    img_file = os.path.join(IMG_PATH, IMAGE_NAME)
    lbl_file = os.path.join(LBL_PATH, IMAGE_NAME.replace(".jpg", ".txt"))

    if not os.path.exists(img_file) or not os.path.exists(lbl_file):
        raise FileNotFoundError("Image or label file not found")

    img = cv2.imread(img_file)
    h, w = img.shape[:2]

    boxes = load_yolo_label(lbl_file, w, h)

    for bbox, poly in boxes:
        # Draw bbox
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Draw polygon mask outline
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)

    out_path = "models/stage2/trials/mask_preview.jpg"
    cv2.imwrite(out_path, img)
    print(f"✅ Saved mask preview to {out_path}")

    if SHOW:
        cv2.imshow("Mask Preview", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize()
