import os
import shutil
import glob
from ultralytics import YOLO # type: ignore
from config.config import Config
from src.utils import patch_torch_load, save_checkpoint_callback

def get_last_checkpoint():
    existing_ckpts = glob.glob(f"{Config.MODEL_OUTPUT_DIR}/last.pt")
    if existing_ckpts:
        return max(existing_ckpts, key=os.path.getctime)
    kaggle_ckpt = f"{Config.WORKING_DIR}/runs/segment/{Config.RUN_NAME}/weights/last.pt"
    if os.path.exists(kaggle_ckpt):
        return kaggle_ckpt
    return None

def train_model():
    patch_torch_load()
    os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)

    last_ckpt = get_last_checkpoint()
    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
        model = YOLO(last_ckpt)
    else:
        print("Starting new training...")
        model = YOLO(Config.BASE_MODEL)

    model.add_callback("on_fit_epoch_end", lambda trainer: save_checkpoint_callback(trainer, Config.MODEL_OUTPUT_DIR))
    train_args = dict(
        data=f"{Config.DATA_DIR}/data.yaml",
        epochs=Config.EPOCHS,
        imgsz=Config.IMG_SIZE,
        batch=Config.BATCH_SIZE,
        project=f"{Config.WORKING_DIR}/runs/segment",
        name=Config.RUN_NAME,
        device=Config.DEVICE,
        optimizer=Config.OPTIMIZER,
        lr0=Config.LR,
        lrf=Config.LRF,
        weight_decay=Config.WEIGHT_DECAY,
        patience=Config.PATIENCE,
        cache=True,
        workers=2,
        close_mosaic=10,
        amp=True,
        mixup=0.1,
        copy_paste=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.9,
        fliplr=0.5,
        max_det=3000,
        conf=0.001,
        iou=0.6
    )

    results = model.train(**train_args)
    print("Training complete!")
    exp_path = f"{Config.WORKING_DIR}/runs/segment/{Config.RUN_NAME}"
    best_pt = f"{exp_path}/weights/best.pt"
    last_pt = f"{exp_path}/weights/last.pt"

    if os.path.exists(best_pt):
        dest_best = f"{Config.MODEL_OUTPUT_DIR}/yolov9_damage_seg_best.pt"
        shutil.copy2(best_pt, dest_best)
        print(f"✅ Best model saved: {dest_best}")

    if os.path.exists(last_pt):
        dest_last = f"{Config.MODEL_OUTPUT_DIR}/yolov9_damage_seg_last.pt"
        shutil.copy2(last_pt, dest_last)
        print(f"Last model saved: {dest_last}")

    print("\nTraining artifacts:")
    print(f"   - Results: {exp_path}/results.png")
    print(f"   - Confusion matrix: {exp_path}/confusion_matrix.png")
    print(f"   - Validation examples: {exp_path}/val_batch*_pred.jpg")

    return results

if __name__ == "__main__":
    train_model()
