from enum import Enum
class MODEL(Enum):
    UNET = 0
    TIRAMISUNET = 1
    REFINED_UNET = 2
    PSPNET2 = 3

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/test-result/'

##------------------- Setup InputSize ----------------------------
INPUT_WIDTH = 512 # 576 # 512
INPUT_HEIGHT = 512 # 768 # 512

##-------------------- Configure #Classes ------------------------
NUM_CLASS = 2 # background / foreground till now

##------------------- Setup Dataset ------------------------------
# train/test dataset for head segmentation
TRAIN_DATASET = "trainSet.txt"
TEST_DATASET = "testSet.txt"

# train/test dataset for portrait segmentation
# TRAIN_DATASET = "trainSet-0.9-v2.3u.txt"
# TEST_DATASET = "testSet-0.9-v2.3u.txt"

## ----------------- Setup Model ----------------------------------
## ----------------- pspnet2 -----------------
# USE_REFINE_NET = False
# MODEL_DIR = "koutou_tf_180321"
# MODEL_TYPE = MODEL.PSPNET2
## ----------------- unet ---------------------
USE_REFINE_NET = False
MODEL_DIR = "koutou_tf_1218"
MODEL_TYPE = MODEL.UNET

# ##----------------- tiramisuNet --------------
# USE_REFINE_NET = False
# MODEL_DIR = "koutou_tf_180123"
# MODEL_TYPE = MODEL.TIRAMISUNET

## ----------------- refinenet -----------------
# USE_REFINE_NET = True
# MODEL_DIR = "koutou_tf_180211"
# MODEL_TYPE = MODEL.REFINED_UNET

## ---------- Configure train + test or test only ------------------
IS_TRAIN = True