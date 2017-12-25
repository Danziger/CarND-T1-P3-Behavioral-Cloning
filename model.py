import cv2
import math

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Convolution2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, Callback

import constants as CONST
import utils


# LOAD SAMPLES:

all_samples = utils.load_samples(CONST.DATA_DIR, CONST.IMG_DIR, CONST.LOG_FILE, [CONST.BEACH_EXAMPLE_FILE])


# TRAIN/VALIDATION SPLITS:

train_samples, validation_samples = train_test_split(all_samples, test_size=CONST.TEST_SIZE)

TRAIN_SAMPLES_COUNT = utils.calculate_augmented_size(train_samples, CONST.ANGLE_CORRECTION, CONST.BEACH_FILTER)
VALIDATION_SAMPLES_COUNT = utils.calculate_augmented_size(validation_samples, CONST.ANGLE_CORRECTION, CONST.SKIP_FILTER)


# print('Train samples = %d, count = %d' % (len(train_samples), TRAIN_SAMPLES_COUNT))

# IMAGE GENERATOR:
# Will not artificially augment validation dat aset with methods other than flipping and using all 3 cameras.

def generator(samples, batch_size=CONST.GENERATOR_BATCH_SIZE, filter=CONST.SKIP_FILTER):
    num_samples = len(samples)

    # TODO Update this?

    while True: # Loop forever so the generator never terminates

        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                utils.get_samples(images, angles, batch_sample, CONST.ANGLE_CORRECTION, filter)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


# DATASETS GENERATORS:

train_generator = generator(train_samples, batch_size=CONST.GENERATOR_BATCH_SIZE, filter=CONST.BEACH_FILTER)
validation_generator = generator(validation_samples, batch_size=CONST.GENERATOR_BATCH_SIZE)


# TEST MODEL:

model = Sequential()

model.add(Cropping2D(cropping=((CONST.CROP_TOP, CONST.CROP_BOTTOM), (0, 0)), input_shape=(CONST.HEIGHT, CONST.WIDTH, CONST.CHANNELS)))

model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Convolution2D(24, 5, 5, W_regularizer=l2(0.001), activation='elu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, W_regularizer=l2(0.001), activation='elu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, W_regularizer=l2(0.001), activation='elu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, W_regularizer=l2(0.001), activation='elu'))
model.add(Convolution2D(64, 3, 3, W_regularizer=l2(0.001), activation='elu'))

model.add(Flatten())

model.add(Dense(100, W_regularizer=l2(0.001), activation='elu'))
model.add(Dense(50, W_regularizer=l2(0.001), activation='elu'))
model.add(Dense(10, W_regularizer=l2(0.001), activation='elu'))
model.add(Dense(1))


# TRAINING THE MODEL:

model.compile(loss='mse', optimizer=Adam(lr=0.0005))

# checkpoint = ModelCheckpoint(CONST.MODEL_PREFIX + '{epoch:03d}.h5')

history = model.fit_generator(
    train_generator,
    samples_per_epoch=TRAIN_SAMPLES_COUNT,
    validation_data=validation_generator,
    nb_val_samples=VALIDATION_SAMPLES_COUNT,
    nb_epoch=CONST.EPOCHS
    # callbacks=[checkpoint]
)


# SAVE:

model.save(CONST.MODEL_FILE)


# MODEL SUMMARY:

print(model.summary())


# PLOT:

print(history.history['loss'])
print(history.history['val_loss'])

# plt.plot()
# plt.plot()
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
