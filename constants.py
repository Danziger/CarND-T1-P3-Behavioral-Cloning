# DATA FILES:

DATA_DIR = './data/'
IMG_DIR = '/IMG/'

LOG_FILE = '/driving_log.csv'
BEACH_4_CLOCK_FILE = 'BEACH-4-LAPS-CLOCKWISE'
BEACH_4_ANTICLOCK_FILE = 'BEACH-4-LAPS-ANTICLOCKWISE'

# IMAGE ATTRIBUTES

CHANNELS = 3
WIDTH = 320
HEIGHT = 160


# DATA SPLIT:

TEST_SIZE = 0.2


# DATA AUGMENTATION:

ANGLE_CORRECTION = 0.25


# DATA PREPROCESSING:

CROP_TOP = 70
CROP_BOTTOM = 25
CROP_SIDES = 0
CROP_WIDTH = WIDTH - 2 * CROP_SIDES
CROP_HEIGHT = HEIGHT - CROP_TOP - CROP_BOTTOM


# GENERATOR:

GENERATOR_BATCH_SIZE = 6


# HYPERPARAMS:

EPOCHS = 16


# MODEL FILE:

MODEL_PREFIX = './MODEL-'
MODEL_EXTENSION = '.h5'
MODEL_FILE = MODEL_PREFIX + 'FINAL' + MODEL_EXTENSION
