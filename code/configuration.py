DEVICE = "gpu"   # cpu or gpu

# some training parameters
EPOCHS = 50 #50
BATCH_SIZE = 16
NUM_CLASSES = 14
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
CHANNELS = 3

save_model_dir = "saved_model/"
save_every_n_epoch = 1
test_image_dir = ""

dataset_dir = "dataset/"
train_dir = "data" #dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"
# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0.2
