from os import listdir, rename, makedirs
from os.path import isdir, join
import random

# TODO: tidy up
# TODO: merge with download_dataset.py


def split_dataset(data_path, validation_path, threshold):
    classes = [d for d in listdir(data_path) if isdir(join(data_path, d))]

    makedirs(validation_path, exist_ok=True)
    for cls_dir in classes:
        for photo in listdir(data_path + cls_dir):
            if random.uniform(0, 100) < threshold:
                makedirs(validation_path + '/' + cls_dir, exist_ok=True)
                rename(data_path + cls_dir + '/' + photo, validation_path + cls_dir + '/' + photo)


if __name__ == "__main__":
    path = "./dataset/"
    threshold = 20
    val_path = "./validation/"
    split_dataset(path, val_path, threshold)

    print("Done.")

