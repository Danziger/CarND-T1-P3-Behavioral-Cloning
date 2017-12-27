import numpy as np
import cv2
import csv
import random
import math


def get_csv_dir(data_dir, session_dir, log_file):
    return data_dir + session_dir + log_file


def get_img_dir(data_dir, session_dir, images_dir, image_file):
    return data_dir + session_dir + images_dir + image_file


# Extract the samples from the CSV as a list of dictionaries:
def load_samples(data_dir, images_dir, log_file, session_dirs):

    samples = []

    for session_dir in session_dirs:
        csv_dir = get_csv_dir(data_dir, session_dir, log_file)

        with open(csv_dir) as file:
            # HEADERS: center, left, right, steering, throttle, brake, speed

            reader = csv.reader(file, skipinitialspace=True)

            samples += [{
                "center": get_img_dir(data_dir, session_dir, images_dir, line[0]),
                "left":  get_img_dir(data_dir, session_dir, images_dir, line[1]),
                "right":  get_img_dir(data_dir, session_dir, images_dir, line[2]),
                "steering": float(line[3]),
            } for line in list(reader)]

    return samples


# Given a sample, extract all 3 images and angles:
def get_samples(images, angles, sample, angle_correction, filter):
    angle = sample["steering"]

    get_sample(images, angles, sample["center"], angle, filter)
    get_sample(images, angles, sample["left"], angle + angle_correction, filter)
    get_sample(images, angles, sample["right"], angle - angle_correction, filter)


# Get a single image, flip it, and augment both of them:
def get_sample(images, angles, name, angle, filter):
    img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2YUV)

    augment_sample(images, angles, img, angle, filter)
    augment_sample(images, angles, np.fliplr(img), -angle, filter)


# Augment the given sample as many times as filter indicates:
def augment_sample(images, angles, base_img, angle, filter):
    times = filter(angle)

    if times:
        angles += [angle] * (times + 1)
        augmented_images = [base_img]

        while times > 0:
            augmented_images.append(augment(base_img))

            times -= 1

        images += augmented_images
    else:
        angles.append(angle)
        images.append(base_img)


# Augment a single image randomly or based on random1 and random2 params:
def augment(base_img, random1=None, random2=None):
    img = base_img[:,:,:]
    h, w = img.shape[0:2]
    m = int(w/2)

    type = random.randint(0, 8) if random1 is None else random1 # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

    # Brightness (or contrast): 50%/50%

    if type <= 1: # 0, 1: All image
        img = contrast(img, 0, w) if type is 0 else brightness(img, 0, w)
    elif type <= 3: # 2, 3: Left
        img = contrast(img, 0, m) if type is 2 else brightness(img, 0, m)
    elif type <= 5: # 4, 5:  Right
        img = contrast(img, m, w) if type is 4 else brightness(img, m, w)
    elif type <= 7: # 6, 7:  Both
        img = contrast(img, 0, m) if type is 6 else brightness(img, 0, m)
        img = contrast(img, m, w) if type is 6 else brightness(img, m, w)
    # else: 8, 9: Nothing here

    # TODO: Update this to have more possibilities?

    # Noise/blur:

    type = random.randint(0, 1 if type > 7 else 2) if random2 is None else random2 # 0, 1, 2

    if type is 0:
        # Noise (sharp)
        # print('Sharp')
        k = random.choice([3, 5, 7, 9])
        img = cv2.addWeighted(img, 2, cv2.GaussianBlur(img, (k, k), 0), -1, 0)
    elif type is 1:
        # Blur
        # print('Blur')
        k = random.choice([3, 5, 7, 9])
        img = cv2.GaussianBlur(img, (k, k), 0)
    # else: 2: Nothing here

    return img


# Add/remove contrast to a YUV image by multiplying it by 0.75 - 1.25 and clip the values to prevent them from
# wrapping (0 - 255):
def contrast(img, start, end):
    value = 0.75 + random.random() * 0.5

    data32 = np.asarray(img, dtype="int32")
    data32[:, start:end, 0] = data32[:, start:end, 0] * value

    np.clip(data32, 0, 255, out=data32)

    return  data32.astype('uint8')


# Add/remove brightness to a YUV image by adding it +/- 16 and clip the values to prevent them from wrapping (0 - 255):
def brightness(img, start, end):
    value = random.randint(-16, 16)
    value = value + 16 if value > 16 else value - 16

    data32 = np.asarray(img, dtype="int32")
    data32[:, start:end, 0] = data32[:, start:end, 0] + value

    np.clip(data32, 0, 255, out=data32)

    return  data32.astype('uint8')


# Calculates the number of angles after using side cameras, mirroring and augmentation:
def calculate_augmented_size(samples, angle_correction, filter):
    angles = list(map(lambda sample: sample["steering"], samples))

    basic_augmented_angles = multiply_angles(angles, angle_correction)
    advanced_augmented_angles = augment_angles(basic_augmented_angles, filter)

    return len(advanced_augmented_angles)


# Calculate the angles after using side cameras and mirroring:
def multiply_angles(angles, angle_correction):
    return angles \
        + list(map(lambda angle: -angle, angles)) \
        + list(map(lambda angle: angle + angle_correction, angles)) \
        + list(map(lambda angle: angle - angle_correction, angles)) \
        + list(map(lambda angle: - (angle + angle_correction), angles)) \
        + list(map(lambda angle: - (angle - angle_correction), angles))


# Calculate the angles after using augmentation based on the filter param:
def augment_angles(angles, filter):
    augmented = angles + []

    for angle in angles:
        times = filter(angle)

        if times:
            augmented += [angle] * times

    return augmented