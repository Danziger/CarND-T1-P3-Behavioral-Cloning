import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Convolution2D, MaxPooling2D, Dropout

import constants as CONST
import utils


# LOAD SAMPLES:

samples = utils.load_samples(CONST.DATA_DIR, CONST.IMG_DIR, CONST.LOG_FILE, [
    CONST.BEACH_4_CLOCK_FILE,
    CONST.BEACH_4_ANTICLOCK_FILE,
])


# TRAIN/VALIDATION SPLITS:

train_samples, validation_samples = train_test_split(samples, test_size=CONST.TEST_SIZE)

TRAIN_SAMPLES_COUNT = len(train_samples) * 6
VALIDATION_SAMPLES_COUNT = len(validation_samples) * 6


# IMAGE GENERATOR:

def generator(samples, batch_size=CONST.GENERATOR_BATCH_SIZE):
    num_samples = len(samples)

    while True: # Loop forever so the generator never terminates

        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                # Names:

                # name_center = CONST.IMG_DIR + batch_sample[0].replace('\\', '/').split('/')[-1]
                # name_left = CONST.IMG_DIR + batch_sample[1].replace('\\', '/').split('/')[-1]
                # name_right = CONST.IMG_DIR + batch_sample[2].replace('\\', '/').split('/')[-1]

                name_center = batch_sample["center"]
                name_left = batch_sample["left"]
                name_right = batch_sample["right"]

                # Center Image:

                image_center = cv2.imread(name_center)
                image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2YUV)
                image_center = image_center[...,::-1]
                angle_center = batch_sample["steering"]
                images.append(image_center)
                angles.append(angle_center)

                image_center_flip = np.fliplr(image_center)
                angle_center_flip = -angle_center
                images.append(image_center_flip)
                angles.append(angle_center_flip)

                # Left Image:

                image_left = cv2.imread(name_left)
                image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2YUV)
                image_left = image_center[...,::-1]
                angle_left = angle_center + CONST.ANGLE_CORRECTION
                images.append(image_left)
                angles.append(angle_left)

                image_left_flip = np.fliplr(image_left)
                angle_left_flip = -angle_left
                images.append(image_left_flip)
                angles.append(angle_left_flip)

                # Right Image:

                image_right = cv2.imread(name_right)
                image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2YUV)
                image_right = image_center[...,::-1]
                angle_right = angle_center - CONST.ANGLE_CORRECTION
                images.append(image_right)
                angles.append(angle_right)

                image_right_flip = np.fliplr(image_right)
                angle_right_flip = -angle_right
                images.append(image_right_flip)
                angles.append(angle_right_flip)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


# DATASETS GENERATORS:

train_generator = generator(train_samples, batch_size=CONST.GENERATOR_BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=CONST.GENERATOR_BATCH_SIZE)


# TEST MODEL:

# TODO: Original one uses YUV instead of RGB. Drive.py uses RGB but here is BGR!
# TODO: Resize images to train faster?
# TODO: Grayscale?
# TODO: ...?

model = Sequential()

model.add(Cropping2D(cropping=((CONST.CROP_TOP, CONST.CROP_BOTTOM), (0, 0)), input_shape=(CONST.HEIGHT, CONST.WIDTH, CONST.CHANNELS)))

model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))

model.add(Flatten())

model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))


# TRAINING THE MODEL:

model.compile(loss='mse', optimizer='adam')

history = model.fit_generator(
    train_generator,
    samples_per_epoch=TRAIN_SAMPLES_COUNT,
    validation_data=validation_generator,
    nb_val_samples=VALIDATION_SAMPLES_COUNT,
    nb_epoch=CONST.EPOCHS
)


# SAVE:

model.save(CONST.MODEL_FILE)


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
