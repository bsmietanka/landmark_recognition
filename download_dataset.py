import csv
import argparse
from os import listdir, rename, makedirs
from os.path import isdir, join
import urllib.request
import random
from multiprocessing.dummy import Pool as ThreadPool


def split_dataset(data_path, validation_path, threshold):
    classes = [d for d in listdir(data_path) if isdir(join(data_path, d))]

    makedirs(validation_path, exist_ok=True)
    for cls_dir in classes:
        for photo in listdir(data_path + cls_dir):
            if random.uniform(0, 100) < threshold:
                makedirs(validation_path + '/' + cls_dir, exist_ok=True)
                rename(data_path + cls_dir + '/' + photo, validation_path + cls_dir + '/' + photo)


class ImageDownloader:
    def __init__(self, path):
        self.directory = path
        self.count = 0
        self.total = 0

    def save_image(self, urlRow):
        if self.count % 100 == 0 and self.count != 0:
            print("Pobrano już " + str(self.count) + "/" + str(self.total)+" obrazków.")
        self.count = self.count+1
        makedirs(self.directory + '/' + urlRow[2], exist_ok=True)
        try:
            urllib.request.urlretrieve(urlRow[1], self.directory + '/' + urlRow[2] + '/' + urlRow[0] + ".jpg")
        except:
            print("404 - File not found")

    def save_images_multithreaded(self, urlData, threadCount=24):
        print("Rozpoczęto pobieranie " + str(len(urlData)) + " obrazów.")
        self.total = len(urlData)
        pool = ThreadPool(threadCount)
        pool.map(self.save_image, urlData)
        pool.close()


def main():
    parser = argparse.ArgumentParser(description='Download images specified in .csv file')
    parser.add_argument('--inputFile', default='csv/dataset_filtered.csv', help='path to input csv data file')
    parser.add_argument('--images', default="dataset", help='Where to save training images')
    parser.add_argument('--validation', default="validation", help='Where to save validation images')
    parser.add_argument('--split', default="20", help='Percentage of images to be used as validation data')
    parser.add_argument('--threads', default="24", help='How many threads to use')
    args = parser.parse_args()

    print("Rozpoczęto działanie skryptu")
    urlData = list()
    with open(args.inputFile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, row in enumerate(reader):
            urlData.append(row)

    imDownloader = ImageDownloader(args.images)
    imDownloader.save_images_multithreaded(urlData, int(args.threads))
    split_dataset(args.images+'/', args.validation+'/', int(args.split))

    print("Pobrano zdjęcia")


if __name__ == "__main__":
    main()