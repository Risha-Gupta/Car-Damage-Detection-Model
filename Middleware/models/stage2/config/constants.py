INPUT_SIZE = (512, 512)
CHANNELS = 3
BBOX_FORMAT = "pascal_voc"

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

AUGMENTATION_CONFIG = {
    'horizontal_flip': 0.5,
    'rotation_limit': 15,
    'brightness_limit': 0.3,
    'gaussian_noise': 0.3,
    'target_size': INPUT_SIZE
}

MODEL_NAME = "YOLOv8-Medium"
PRETRAINED = True
NUM_CLASSES = 1

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_MAX_NORM = 1.0

METRICS_DECIMAL_PLACES = 4

DATA_DIR = r"C:\Users\Risha\Downloads\CarDD_release\CarDD_release\CarDD_SOD"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
SPLIT_DATA_DIR = f"{DATA_DIR}/split"
CHECKPOINT_DIR = "Middleware/models/stage2/models/checkpoints"
LOG_DIR = "Middleware/models/stage2/logs"
REPORT_OUTPUT = "Middleware/models/stage2/Damage_Localization_Report.xlsx"

LOG_INTERVAL = 10
PROFILE_INFERENCE = True
SAVE_BEST_MODEL = True

STAGE1_MODEL_PATH = "Middleware/models/stage1/mobilenetv3_canny_best.keras"
STAGE1_DAMAGE_THRESHOLD = 0.5
