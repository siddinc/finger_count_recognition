import os


FRAME_DIM = (640, 480)
IMG_DIM = (128, 128)
INPUT_DIM = (1, 128, 128, 1)
TOP, RIGHT, BOTTOM, LEFT = 70, 350, 285, 565
NO_OF_EPOCHS = 20
BATCH_SIZE = 180
TRAIN_PATH = os.path.abspath('../datasets/fingers/train')
TEST_PATH = os.path.abspath('../datasets/fingers/test')
SAVE_MODEL_PATH = os.path.abspath('../models')
LOAD_MODEL_PATH = os.path.abspath('../models')
LABELS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5
}
