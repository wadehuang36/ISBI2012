import os
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiff', dest="tiffFile", type=str)
    parser.add_argument('--output', dest="outFile", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.tiffFile is None:
        files = filter(lambda x: x.endswith(".tif") or x.endswith(".tiff"), os.listdir("./data/ISBI2012"))
        print("The following are tiff files in data folder, choose one that you want to process")
        for i in range(len(files)):
            print("%s ) %s" % (i + 1, files[i]))

        index = raw_input("Enter your name: ")
