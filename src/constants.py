import os


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