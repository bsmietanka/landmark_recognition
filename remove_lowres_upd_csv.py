import csv
import argparse
import os
import os.path
import urllib.request
import datetime
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool


def check_images(input_file, output_file, images, minX, maxX, minY, maxY):
    # Sprawdzam które zdjęcia tak naprawdę są w folderze
    url_data = list()
    with open(images + "/" + input_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, row in enumerate(reader):
            photopath = images+"/"+row[2]+"/"+row[0]+".jpg"
            if os.path.isfile(photopath):
                im = Image.open(photopath)
                p, q = im.size
                x = max(p, q)
                y = min(p, q)
                if (minX <= x <= maxX) and (minY <= y <= maxY):
                    url_data.append(row)

    with open(images + "/" + output_file, 'w', newline='') as outfile:
        outwriter = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for row in url_data:
            if row is not None:
                outwriter.writerow([row[0], row[1], int(row[2])])


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--inputFile', default='trainFiltered.csv', help='path to input data file')
    parser.add_argument('--minX', default='800', help='min value of longer dimension in px')
    parser.add_argument('--maxX', default='1600', help='max value of longer dimension in px')
    parser.add_argument('--minY', default='600', help='min value of shorter dimension in px')
    parser.add_argument('--maxY', default='1200', help='max value of shorter dimension in px')
    parser.add_argument('--outputFile', default='trainFiltered2.csv', help='path to output data file')
    parser.add_argument('--images', default="dataset", help='Folder where images are stored')
    args = parser.parse_args()

    print("Rozpoczęto działanie skryptu")
    # Sprawdzam które zdjęcia tak naprawdę są w folderze
    check_images(args.inputFile, args.outputFile, args.images, int(args.minX), int(args.maxX),
                 int(args.minY), int(args.maxY))
    print("Zakończono pracę skryptu")


if __name__ == "__main__":
    main()