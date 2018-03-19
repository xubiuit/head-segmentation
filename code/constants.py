INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/test-result/'

##------------------------------------------------
INPUT_WIDTH = 512 # 576 # 512
INPUT_HEIGHT = 512 # 768 # 512

##-------------------------------------------------
# train/test dataset for head segmentation
TRAIN_DATASET = "trainSet.txt"
TEST_DATASET = "testSet.txt"

# train/test dataset for portrait segmentation
# TRAIN_DATASET = "trainSet-0.9-v2.3u.txt"
# TEST_DATASET = "testSet-0.9-v2.3u.txt"

## ----------------- test unet ---------------------
USE_REFINE_NET = False
MODEL_DIR = "koutou_tf_1218"

# ##----------------- test tiramisuNet --------------
# USE_REFINE_NET = False
# MODEL_DIR = "koutou_tf_180123"

## ----------------- test unet + refinenet ---------
# USE_REFINE_NET = True
# MODEL_DIR = "koutou_tf_180211"

## -------------------------------------------------
IS_TRAIN = False