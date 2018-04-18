from os import listdir, rename
from os.path import isdir, join
import random

if __name__ == "__main__":
    path = "./dataset/"
    threshold = 20
    validation_path = "./validation/"

    classes = [d for d in listdir(path) if isdir(join(path, d))]
    
    for cls_dir in classes:
        for photo in listdir(path + cls_dir):
            if random.uniform(0,100) < threshold:
                rename(join(path, cls_dir, photo), join(validation_path, cls_dir, photo))
    
    print("Done.")

