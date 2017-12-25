import numpy as np
import cv2
import csv
import random
import math


def get_csv_dir(data_dir, session_dir, log_file):
    return data_dir + session_dir + log_file


def get_img_dir(data_dir, session_dir, images_dir, image_file):
    return data_dir + session_dir + images_dir + image_file


def load_samples(data_dir, images_dir, log_file, session_dirs):

    samples = []

    for session_dir in session_dirs:
        csv_dir = get_csv_dir(data_dir, session_dir, log_file)

        with open(csv_dir) as file:
            # center, left, right, steering, throttle, brake, speed

            reader = csv.reader(file, skipinitialspace=True)

            samples += [{
                "center": get_img_dir(data_dir, session_dir, images_dir, line[0]),
                "left":  get_img_dir(data_dir, session_dir, images_dir, line[1]),
                "right":  get_img_dir(data_dir, session_dir, images_dir, line[2]),
                "steering": float(line[3]),
            } for line in list(reader)]

    return samples


def get_samples(images, angles, sample, angle_correction, filter):
    angle = sample["steering"]

    get_sample(images, angles, sample["center"], angle, filter)
    get_sample(images, angles, sample["left"], angle + angle_correction, filter)
    get_sample(images, angles, sample["right"], angle - angle_correction, filter)


def get_sample(images, angles, name, angle, filter):
    img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2YUV)

    augment_samples(images, angles, img, angle, filter)
    augment_samples(images, angles, np.fliplr(img), -angle, filter)


def augment_samples(images, angles, base_img, angle, filter):
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


def augment(base_img): # TODO: Pass options to force type values!
    img = base_img[:,:,:]
    h, w = img.shape[0:2]
    m = int(w/2)

    type = random.randint(0, 8) # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

    # TODO: Add a file to see examples with my grid!
    # TODO: Finish this and plot examples! Get grid helper function from my other repos!

    # Brightness (or contrast): 50%/50%

    if type <= 1: # 0, 1: All image
        img = contrast(img, 0, w) if type is 0 else brigtness(img, 0, w)
    elif type <= 3: # 2, 3: Left
        img = contrast(img, 0, m) if type is 2 else brigtness(img, 0, m)
    elif type <= 5: # 4, 5:  Right
        img = contrast(img, m, w) if type is 4 else brigtness(img, m, w)
    elif type <= 7: # 6, 7:  Both
        img = contrast(img, 0, m) if type is 6 else brigtness(img, 0, m)
        img = contrast(img, m, w) if type is 6 else brigtness(img, m, w)
    # else: 8, 9: Nothing here

    # Noise/blur:

    type = random.randint(0, 1 if type > 7 else 2) # 0, 1, 2

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


def contrast(img, start, end):
    value = 0.75 + random.random() * 0.5

    # print('Contrast = %f, start = %d, end = %d' % (value, start, end))

    data32 = np.asarray(img, dtype="int32")
    data32[:, start:end, 0] = data32[:, start:end, 0] * value

    np.clip(data32, 0, 255, out=data32)

    return  data32.astype('uint8')


def brigtness(img, start, end):
    value = random.randint(-16, 16)
    value = value + 16 if value > 16 else value - 16

    # print('Value = %d, start = %d, end = %d' % (value, start, end))

    data32 = np.asarray(img, dtype="int32")
    data32[:, start:end, 0] = data32[:, start:end, 0] + value

    np.clip(data32, 0, 255, out=data32)

    return  data32.astype('uint8')


def calculate_augmented_size(samples, angle_correction, filter):
    angles = list(map(lambda sample: sample["steering"], samples))

    # CALCULATE DISTRIBUTION AFTER USING SIDE CAMERAS AND MIRRORING AUGMENTATION:

    basic_augmented_angles = angles \
                      + list(map(lambda angle: -angle, angles)) \
                      + list(map(lambda angle: angle + angle_correction, angles)) \
                      + list(map(lambda angle: angle - angle_correction, angles)) \
                      + list(map(lambda angle: - (angle + angle_correction), angles)) \
                      + list(map(lambda angle: - (angle - angle_correction), angles))

    advanced_augmented_angles = basic_augmented_angles + []

    for angle in basic_augmented_angles:
        times = filter(angle)

        if times:
            advanced_augmented_angles += [angle] * times

    return len(advanced_augmented_angles)
