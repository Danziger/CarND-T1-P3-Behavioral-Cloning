import matplotlib.pyplot as plt

import constants as CONST
import utils

samples = utils.load_samples(CONST.DATA_DIR, CONST.IMG_DIR, CONST.LOG_FILE, [
    CONST.BEACH_4_ANTICLOCK_FILE,
    CONST.BEACH_4_CLOCK_FILE,
    CONST.RECO_BEACH_1_ANTICLOCK_FILE,
    CONST.RECO_BEACH_1_CLOCK_FILE,
    CONST.MOUNTAIN_4_ANTICLOCK_FILE,
    CONST.MOUNTAIN_4_CLOCK_FILE,
    CONST.RECO_MOUNTAIN_1_ANTICLOCK_FILE,
    CONST.RECO_MOUNTAIN_1_CLOCK_FILE
])

all = []

angles = list(map(lambda sample: sample["steering"], samples))

all.extend(angles)
all.extend(map(lambda angle: -angle, angles))

all.extend(list(map(lambda angle: angle + 0.25, angles)))
all.extend(list(map(lambda angle: angle - 0.25, angles)))
all.extend(list(map(lambda angle: - (angle + 0.25), angles)))
all.extend(list(map(lambda angle: - (angle - 0.25), angles)))

plt.hist(all)
plt.show()

# TODO: Add plots for each track, both tracks, each track recovery and both track recovery, 6 in total