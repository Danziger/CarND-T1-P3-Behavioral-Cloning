import cv2
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Convolution2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils.visualize_util import plot

import utils
import constants as CONST


# LOAD SAMPLES:
# Load all samples from one or multiple files.

all_samples = utils.load_samples(CONST.DATA_DIR, CONST.IMG_DIR, CONST.LOG_FILE, CONST.BEACH_DATA_FILES)


# TRAIN/VALIDATION SPLITS:
# Split all the samples in training and validation and calculate the number of samples in each set taking into account
# the criteria and methods that will be used to augment them.

train_samples, validation_samples = train_test_split(all_samples, test_size=CONST.TEST_SIZE)

TRAIN_SAMPLES_COUNT = utils.calculate_augmented_size(train_samples, CONST.ANGLE_CORRECTION, CONST.SKIP_FILTER)
VALIDATION_SAMPLES_COUNT = utils.calculate_augmented_size(validation_samples, CONST.ANGLE_CORRECTION, CONST.SKIP_FILTER)


# IMAGE GENERATOR:
# Will not artificially augment validation dataset with methods other than flipping and using all 3 cameras, unless a
# filter param other than CONST.SKIP_FILTER is provided.

def generator(samples, batch_size=CONST.GENERATOR_BATCH_SIZE, filter=CONST.SKIP_FILTER):
    num_samples = len(samples)

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

            # When using augmentation, it will yield batches of different size:
            yield shuffle(X_train, y_train)


# DATASETS GENERATORS:
# Create both generators. The validation one will use the default, basic augmentation.

train_generator = generator(train_samples, filter=CONST.SKIP_FILTER)
validation_generator = generator(validation_samples)


# TEST MODEL:
# Implement the model from NVIDIA with 2 additional Cropping2D and Lambda layers to remove unnecessary portions of the
# image and normalize the data. It also adds L2 regularization to all the layers and elu activations.

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
# Train the model using Adam optimizer with an initial learning rate that is half the default value (0.001).

model.compile(loss='mse', optimizer=Adam(lr=0.0005))

# Save a model for each epochs:
# checkpoint = ModelCheckpoint(CONST.MODEL_PREFIX + '{epoch:03d}.h5')

# Start training using the generators:
history = model.fit_generator(
    train_generator,
    samples_per_epoch=TRAIN_SAMPLES_COUNT,
    validation_data=validation_generator,
    nb_val_samples=VALIDATION_SAMPLES_COUNT,
    nb_epoch=CONST.EPOCHS
    # callbacks=[checkpoint]
)


# SAVE:
# Save the final model:

model.save(CONST.MODEL_FILE)


# MODEL INFO:
# Print a summary of the model and save a diagram of it as an image:

print(model.summary())

plot(model, to_file=CONST.MODEL_DIAGRAM_FILE, show_shapes=True, show_layer_names=False)


# PLOT:
# Print history loss and validation loss, as it can't be plotted on AWS directly.

print(history.history['loss'])
print(history.history['val_loss'])
