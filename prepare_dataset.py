import csv
import argparse
import os
import urllib.request
import datetime
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
from operator import is_not
from functools import partial


class UrlChecker:
    def __init__(self):
        self.counter = 0

    def exists(self, data_url):
        nazwa = data_url[0]
        url = data_url[1]
        kategoria = data_url[2]
        self.counter += 1
        if self.counter%250 == 0:
            print("Sprawdzono " + str(self.counter))
        try:
            urllib.request.urlopen(url)
        except:
            return None
        else:
            return nazwa, url, kategoria

    def checkUrls(self, data, threads_number=24):
        pool = ThreadPool(threads_number)
        foty = pool.map(self.exists, data)
        return foty

    def filterDeadUrls(self, data):
        return [x for x in data if x is not None]


def main():
    parser = argparse.ArgumentParser(description='Script that chooses selected number of classes containing selected '
                                                 'number of images of all sizes from dataset')
    parser.add_argument('--inputFile', default='train.csv', help='path to input data file')
    parser.add_argument('--outputFile', default='trainFiltered.csv', help='path to output data file')
    parser.add_argument('--imNum', default='10000', help='max number of pictures of all sizes in selected classes')
    parser.add_argument('--classes', default='100', help='number of selected classes')
    parser.add_argument('--images', default="csv", help='Where to save the new .csv file')
    args = parser.parse_args()

    print("Rozpoczęto działanie skryptu")
    classes = dict()
    # Liczę ile jest fotek w każdej klasie
    with open(args.inputFile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if int(row[2]) in classes:
                classes[int(row[2])] = classes[int(row[2])] + 1
            else:
                classes[int(row[2])] = 1

    # Wybieram klasy mające najwięcej zdjęć tak, żeby nie przekroczyć w sumie max
    max = int(args.imNum)
    min_number_of_classes = int(args.classes)
    so_far = 0
    sorted_classes = sorted(classes.items(), key=lambda tup: tup[1], reverse=True)
    classes_to_stay = set()
    for i, row in enumerate(sorted_classes):
        if so_far > max:
            break
        if row[1] > max/min_number_of_classes:
            continue
        else:
            classes_to_stay.add(row[0])
            so_far += row[1]
    print("Wybrano odpowiednie klasy")

    # Tworzę nowy plik .csv zawierający tylko wybrane klasy
    url_data = list()
    with open(args.inputFile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if int(row[2]) in classes_to_stay:
                url_data.append(row)

    checker = UrlChecker()
    url_data = checker.checkUrls(url_data)
    url_data = checker.filterDeadUrls(url_data)
    url_data = sorted(url_data, key=lambda tup: int(tup[2]))

    with open(args.images+"/"+args.outputFile, 'w', newline='') as outfile:
        outwriter = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for row in url_data:
            if row is not None:
                outwriter.writerow([row[0], row[1], int(row[2])])


if __name__ == "__main__":
    main()