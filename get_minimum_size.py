from os import walk
from os.path import join
from PIL import Image
import sys

def get_min_size(path):
    min_size = {'height' : sys.maxsize, 'width' : sys.maxsize}
    for (dirpath, _, filenames) in walk(path):
        for file_path in filenames:
            im = Image.open(join(dirpath, file_path))
            (width, height) = im.size
            if width < min_size['width']:
                min_size['width'] = width
            if height < min_size['height']:
                min_size['height'] = height
    return min_size

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print( f'Size : {get_min_size(sys.argv[1])}' )