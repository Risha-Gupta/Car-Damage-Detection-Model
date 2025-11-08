INPUT_SIZE = (512, 512)
CHANNELS = 3
BBOX_FORMAT = "coco"  # Changed from pascal_voc to COCO format

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

MODEL_NAME = "Faster R-CNN ResNet50"
FRAMEWORK = "Detectron2"
PRETRAINED = True
NUM_CLASSES = 6  # dent, scratch, crack, glass shatter, lamp broken, tire flat

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001  # L2 regularization
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_MAX_NORM = 1.0

DROPOUT_RATE = 0.3
L2_LAMBDA = 0.001

# K-fold cross-validation
K_FOLD_SPLITS = 5

METRICS_DECIMAL_PLACES = 4

DATA_DIR = r"/kaggle/input/car-dd/CarDD_release/CarDD_COCO"
TRAIN_ANNOTATION_FILE = f"{DATA_DIR}/annotations/instances_train2017.json"
VAL_ANNOTATION_FILE = f"{DATA_DIR}/annotations/instances_val2017.json"
TEST_ANNOTATION_FILE = f"{DATA_DIR}/annotations/instances_test2017.json"

TRAIN_IMAGE_DIR = f"{DATA_DIR}/train2017"
VAL_IMAGE_DIR = f"{DATA_DIR}/val2017"
TEST_IMAGE_DIR = f"{DATA_DIR}/test2017"

CHECKPOINT_DIR = "/kaggle/working/checkpoints"
LOG_DIR = "/kaggle/working/logs"
REPORT_OUTPUT = "/kaggle/working/Damage_Detection_Report.xlsx"
PERFORMANCE_TABLE = "/kaggle/working/performance_table.csv"

LOG_INTERVAL = 10
PROFILE_INFERENCE = True
SAVE_BEST_MODEL = True

# Damage class mapping
DAMAGE_CLASSES = {
    0: 'dent',
    1: 'scratch',
    2: 'crack',
    3: 'glass_shatter',
    4: 'lamp_broken',
    5: 'tire_flat'
}

STAGE1_MODEL_PATH = "Middleware/models/stage1/mobilenetv3_canny_best.keras"
STAGE1_DAMAGE_THRESHOLD = 0.5
