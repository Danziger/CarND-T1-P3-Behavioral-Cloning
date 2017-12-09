import csv


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
