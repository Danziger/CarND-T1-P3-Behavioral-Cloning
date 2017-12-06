import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Convolution2D, MaxPooling2D


# CONSTANTS:

# Data:
DATA_DIR = './data/BEACH-4-LAPS/'
IMG_DIR = DATA_DIR + 'IMG/'
LOG_FILE = DATA_DIR + 'driving_log.csv'

TEST_SIZE = 0.2
MODEL_FILE = './model.h5'

# Generator:
GENERATOR_BATCH_SIZE = 32

# Image attributes:
CHANNELS = 3
WIDTH = 320
HEIGHT = 160

# Cropping:
CROP_TOP = 60
CROP_BOTTOM = 20
CROP_SIDES = 0
CROP_WIDTH = WIDTH - 2 * CROP_SIDES
CROP_HEIGHT = HEIGHT - CROP_TOP - CROP_BOTTOM

# LOAD SAMPLES:

with open(LOG_FILE) as csvfile:
    reader = csv.reader(csvfile)

    next(reader)  # Skip the headers.

    SAMPLES = list(reader)


# TRAIN/VALIDATION SPLITS:

train_samples, validation_samples = train_test_split(SAMPLES, test_size=TEST_SIZE)


# IMAGE GENERATOR:

def generator(samples, batch_size=GENERATOR_BATCH_SIZE):
    num_samples = len(samples)

    # TODO: Use also side-camera images with correction...

    while True: # Loop forever so the generator never terminates
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                name = IMG_DIR + batch_sample[0].replace('\\', '/').split('/')[-1]

                image_center = cv2.imread(name)
                angle_center = float(batch_sample[3])
                images.append(image_center)
                angles.append(angle_center)

                image_center_flip = np.fliplr(image_center)
                angle_center_flip = -angle_center
                images.append(image_center_flip)
                angles.append(angle_center_flip)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


# DATASETS GENERATORS:

train_generator = generator(train_samples, batch_size=GENERATOR_BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=GENERATOR_BATCH_SIZE)


# TEST MODEL:

model = Sequential()
model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0, 0)), input_shape=(HEIGHT, WIDTH, CHANNELS)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())

# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())

model.add(Flatten())
# model.add(Dense(60))
model.add(Dense(1))


# TRAINING THE MODEL:

model.compile(loss='mse', optimizer='adam')

history = model.fit_generator(
    train_generator,
    samples_per_epoch=2 * len(train_samples),
    validation_data=validation_generator,
    nb_val_samples=2 * len(validation_samples),
    nb_epoch=7
)


# SAVE:

model.save(MODEL_FILE)


# MODEL SUMMARY:

print(model.summary())


# PLOT:

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
