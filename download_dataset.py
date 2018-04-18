import csv
import argparse
import os
import urllib.request
from multiprocessing.dummy import Pool as ThreadPool

# TODO: change dataset directory structure
#   dataset/
#     |--> train/
#     |    |--> {classes}/
#     |--> test/
#          |--> {classes}/

class ImageDownloader:
    def __init__(self, path):
        self.directory = path
        self.count = 0
        self.total = 0

    def save_image(self, urlRow):
        if self.count % 100 == 0:
            print("Pobrano już " + str(self.count) + "/" + str(self.total)+" obrazków.")
        self.count = self.count+1
        os.makedirs(self.directory + '/' + urlRow[2], exist_ok=True)
        try:
            urllib.request.urlretrieve(urlRow[1], self.directory + '/' + urlRow[2] + '/' + urlRow[0] + ".jpg")
        except:
            print("404 - File not found")

    def save_images_multithreaded(self, urlData, threadCount=24):
        print("Rozpoczęto pobieranie " + str(len(urlData)) + " obrazów.")
        self.total = len(urlData)
        pool = ThreadPool(threadCount)
        pool.map(self.save_image, urlData)


def main():
    parser = argparse.ArgumentParser(description='Download images specified in .csv file')
    parser.add_argument('--inputFile', default='csv/dataset_filtered.csv', help='path to input csv data file')
    parser.add_argument('--images', default="dataset", help='Where to save images')
    parser.add_argument('--threads', default="4", help='How many threads to use')
    args = parser.parse_args()

    print("Rozpoczęto działanie skryptu")
    urlData = list()
    with open(args.inputFile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, row in enumerate(reader):
            urlData.append(row)

    imDownloader = ImageDownloader(args.images)
    imDownloader.save_images_multithreaded(urlData, int(args.threads))
    print("Pobrano zdjęcia")


if __name__ == "__main__":
    main()