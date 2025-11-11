import os
from pathlib import Path

class Config:
    IS_KAGGLE = os.path.exists("/kaggle/input")

    if IS_KAGGLE:
        DATA_DIR = "/kaggle/input/car-yolo-dataset/data/damage_seg"
        WORKING_DIR = "/kaggle/working"
        MODEL_OUTPUT_DIR = f"{WORKING_DIR}/models"
    else:
        PROJECT_ROOT = Path(__file__).parent.parent
        DATA_DIR = PROJECT_ROOT / "data" / "damage_seg"
        WORKING_DIR = PROJECT_ROOT
        MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"

    RUN_NAME = "damage_det_stage3_kaggle"
    EPOCHS = 50
    IMG_SIZE = 640
    BATCH_SIZE = 8
    BASE_MODEL = "yolov9c-seg.pt"
    OPTIMIZER = "AdamW"
    LR = 5e-4
    LRF = 0.01
    WEIGHT_DECAY = 5e-4
    PATIENCE = 25
    DEVICE = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
