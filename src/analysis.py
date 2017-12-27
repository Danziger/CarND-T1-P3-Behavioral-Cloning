import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split

import utils
import constants as CONST


# LOAD DATA:

# Track 1 (Beach):

samples_beach = utils.load_samples(CONST.DATA_DIR, CONST.IMG_DIR, CONST.LOG_FILE, [
    CONST.BEACH_4_ANTICLOCK_FILE,
    CONST.BEACH_4_CLOCK_FILE,
    CONST.RECO_BEACH_1_ANTICLOCK_FILE,
    CONST.RECO_BEACH_1_CLOCK_FILE
])

# Track 2 (Mountain):

samples_mountain = utils.load_samples(CONST.DATA_DIR, CONST.IMG_DIR, CONST.LOG_FILE, [
    CONST.MOUNTAIN_4_ANTICLOCK_FILE,
    CONST.MOUNTAIN_4_CLOCK_FILE,
    CONST.RECO_MOUNTAIN_1_ANTICLOCK_FILE,
    CONST.RECO_MOUNTAIN_1_CLOCK_FILE
])


# EXTRACT BASE ANGLES:

angles_beach = list(map(lambda sample: sample["steering"], samples_beach))
angles_mountain = list(map(lambda sample: sample["steering"], samples_mountain))
angles_all = angles_beach + angles_mountain

# Plot them:

plt.hist([angles_beach, angles_mountain, angles_all])
plt.legend(['Beach', 'Mountain', 'Both'], loc='upper right')
plt.show()


# CALCULATE DISTRIBUTION AFTER USING SIDE CAMERAS AND MIRRORING AUGMENTATION:

angles_beach_6x = utils.multiply_angles(angles_beach, CONST.ANGLE_CORRECTION)
angles_mountain_6x = utils.multiply_angles(angles_mountain, CONST.ANGLE_CORRECTION)
angles_all_6x = angles_beach_6x + angles_mountain_6x

# Plot them:

plt.hist([angles_beach_6x, angles_mountain_6x, angles_all_6x])
plt.legend(['Beach 6x', 'Mountain 6x', 'Both 6x'], loc='upper right')
plt.show()


# CALCULATE DISTRIBUTION AFTER USING IMAGE AUGMENTATION METHODS:

angles_beach_augmented = utils.augment_angles(angles_beach_6x, CONST.BEACH_FILTER)
angles_mountain_augmented = utils.augment_angles(angles_mountain_6x, CONST.MOUNTAIN_FILTER)
angles_all_augmented = angles_beach_augmented + angles_mountain_augmented

# Plot them:

plt.hist([angles_beach_augmented, angles_mountain_augmented, angles_all_augmented])
plt.legend(['Beach Aug.', 'Mountain Aug.', 'Both Aug.'], loc='upper right')
plt.show()


# PLOT ALL:

plt.hist([
    angles_beach,
    angles_beach_6x,
    angles_beach_augmented,

    angles_mountain,
    angles_mountain_6x,
    angles_mountain_augmented,

    angles_all,
    angles_all_6x,
    angles_all_augmented
])

plt.legend([
    'Beach',
    'Beach 6x',
    'Beach Aug.',

    'Mountain',
    'Mountain 6x',
    'Mountain Aug.',

    'Both',
    'Both 6x',
    'Both Aug.'
], loc='upper right')

plt.show()
