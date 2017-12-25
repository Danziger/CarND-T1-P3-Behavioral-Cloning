import math


# DATA FILES:

DATA_DIR = './data/'
IMG_DIR = '/IMG/'

LOG_FILE = '/driving_log.csv'

# Example:

BEACH_EXAMPLE_FILE = 'BEACH-EXAMPLE'

# Beach:

BEACH_4_ANTICLOCK_FILE = 'BEACH-4-LAPS-ANTICLOCKWISE'
BEACH_4_CLOCK_FILE = 'BEACH-4-LAPS-CLOCKWISE'
RECO_BEACH_1_ANTICLOCK_FILE = 'RECOVERY-BEACH-1-LAP-ANTICLOCKWISE'
RECO_BEACH_1_CLOCK_FILE = 'RECOVERY-BEACH-1-LAP-CLOCKWISE'

BEACH_DATA_FILES = [
    BEACH_4_ANTICLOCK_FILE,
    BEACH_4_CLOCK_FILE,
    RECO_BEACH_1_ANTICLOCK_FILE,
    RECO_BEACH_1_CLOCK_FILE
]

# Mountain:

MOUNTAIN_4_ANTICLOCK_FILE = 'MOUNTAIN-4-LAPS-ANTICLOCKWISE'
MOUNTAIN_4_CLOCK_FILE = 'MOUNTAIN-4-LAPS-CLOCKWISE'
RECO_MOUNTAIN_1_ANTICLOCK_FILE = 'RECOVERY-MOUNTAIN-1-LAP-ANTICLOCKWISE'
RECO_MOUNTAIN_1_CLOCK_FILE = 'RECOVERY-MOUNTAIN-1-LAP-CLOCKWISE'

MOUNTAIN_DATA_FILES = [
    MOUNTAIN_4_ANTICLOCK_FILE,
    MOUNTAIN_4_CLOCK_FILE,
    RECO_MOUNTAIN_1_ANTICLOCK_FILE,
    RECO_MOUNTAIN_1_CLOCK_FILE
]

# Both:

ALL_DATA_FILES = BEACH_DATA_FILES + MOUNTAIN_DATA_FILES


# IMAGE ATTRIBUTES

CHANNELS = 3
WIDTH = 320
HEIGHT = 160


# DATA SPLIT:

TEST_SIZE = 0.3


# DATA AUGMENTATION:

ANGLE_CORRECTION = 0.25


# DATA PREPROCESSING:

CROP_TOP = 70
CROP_BOTTOM = 25
CROP_SIDES = 0
CROP_WIDTH = WIDTH - 2 * CROP_SIDES
CROP_HEIGHT = HEIGHT - CROP_TOP - CROP_BOTTOM


# GENERATOR:

GENERATOR_BATCH_SIZE = 30 # Outputs batches of 30 * 6 = 180 images (augmented)


# HYPERPARAMS:

EPOCHS = 24


# MODEL FILE:

MODEL_PREFIX = './model-'
MODEL_EXTENSION = '.h5'
MODEL_FILE = './model-both-pre-analysis.h5'


def BEACH_FILTER(angle):
    times = math.ceil(10 * abs(angle))

    return times if times > 5 else False

def SKIP_FILTER(angle):
    return False
