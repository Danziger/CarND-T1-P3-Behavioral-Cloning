import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split

import constants as CONST
import utils


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


# EXTRACT ANGLES:

angles_beach = list(map(lambda sample: sample["steering"], samples_beach))
angles_mountain = list(map(lambda sample: sample["steering"], samples_mountain))
angles_all = angles_beach + angles_mountain


# PLOT:

plt.hist([angles_beach, angles_mountain, angles_all])
plt.legend(['Beach', 'Mountain', 'Both'], loc='upper right')
plt.show()


# CALCULATE DISTRIBUTION AFTER USING SIDE CAMERAS AND MIRRORING AUGMENTATION:

angles_beach_augmented = angles_beach \
                         + list(map(lambda angle: -angle, angles_beach)) \
                         + list(map(lambda angle: angle + 0.25, angles_beach)) \
                         + list(map(lambda angle: angle - 0.25, angles_beach)) \
                         + list(map(lambda angle: - (angle + 0.25), angles_beach)) \
                         + list(map(lambda angle: - (angle - 0.25), angles_beach))

super_beach = angles_beach_augmented + []

for angle in angles_beach_augmented:
    times = math.ceil(10 * abs(angle))

    if times > 5:
        super_beach += [angle] * times


print('Super Beach # = %d' % len(super_beach))


angles_mountain_augmented = angles_mountain \
                         + list(map(lambda angle: -angle, angles_mountain)) \
                         + list(map(lambda angle: angle + 0.25, angles_mountain)) \
                         + list(map(lambda angle: angle - 0.25, angles_mountain)) \
                         + list(map(lambda angle: - (angle + 0.25), angles_mountain)) \
                         + list(map(lambda angle: - (angle - 0.25), angles_mountain))

super_mountain = angles_mountain_augmented + []

for angle in angles_mountain_augmented:
    times = math.ceil(5 * abs(angle))

    if times > 5:
        super_mountain += [angle] * times

angles_all_augmented = angles_beach_augmented + angles_mountain_augmented
super_all = super_beach + super_mountain

# PLOT:

plt.hist([angles_beach, angles_beach_augmented, super_beach, angles_mountain, angles_mountain_augmented, super_mountain, angles_all, angles_all_augmented, super_all])
plt.legend(['Beach', 'Beach Augmented', 'Super Beach', 'Mountain', 'Mountain Augmented', 'Super Mountain', 'Both', 'Both Augmented', 'Super Both'], loc='upper right')
plt.show()
