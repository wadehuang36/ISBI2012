"""
 Every Data Sets for Caffe needs to be converted to in LevelDB or LMDB (id, data, label)
 Also, ISBI 2012 Data Set needs to be pre-processed in order to be written into DB
 1. mirror images edges with 32 px, so a image becomes 572 * 572.
 2. separate every pixel to 65 * 65.

 So sharp is 30(number) * 262144(512*512) = 7864320 * 65 * 65 (no channel)

 And 25 images for train, 5 images for testing.
 (test-volume.tif can't be used because it don't has labels.
  so we separate train-volume.tif to train and test set)

 Place ensure, these files in ./data
 - train-labels.tif
 - train-volume.tif
"""

import os
import sys
import caffe
import numpy as np
import lmdb
from PIL import Image

DEBUG = False
TRAIN_IMAGES = "./data/train-volume.tif"
TRAIN_LABELS = "./data/train-labels.tif"

TILE_SIZE = 65
TRAIN_DB = "./data/train_" + str(TILE_SIZE)
TEST_DB = "./data/test_" + str(TILE_SIZE)

TRAIN_RANGE = range(0, 25)
TEST_RANGE = range(25, 30)

EDGE_SIZE = int((TILE_SIZE - 1) / 2)


def loadImages(fileName):
    if not os.path.isfile(fileName):
        raise RuntimeError("% is missing." % fileName)

    # read images in tiff format
    images = []
    tiff = Image.open(fileName)
    while True:
        image = np.array(tiff)
        if image.ndim == 2:
            image = image[np.newaxis, ...]

        images.append(image)

        try:
            tiff.seek(tiff.tell() + 1)
        except EOFError:
            # this just means hit end of file (not really an error)
            break

    return np.concatenate(images)


def convertLabels(images):
    labels = np.zeros(images.shape, dtype=int)

    # find classes of labels, if might be 0, 255 or others
    labelClasses = np.sort(np.unique(images))
    for idx, cl in enumerate(labelClasses):
        # change 0 to 0, 255 to 1
        labels[images == cl] = idx

    return labels


def mirrorEdges(images):
    """
    Padding edges by mirror to mirror + 512 + mirror
    :param images:
    :return:
    """
    n, h, w = images.shape

    copy = np.zeros((n, h + TILE_SIZE - 1, w + TILE_SIZE - 1), dtype=images.dtype)
    for ni in range(n):
        # inside is the same
        copy[ni, EDGE_SIZE:h + EDGE_SIZE, EDGE_SIZE:w + EDGE_SIZE] = images[ni, ...]

        # left edge
        copy[ni, :, 0:EDGE_SIZE] = np.fliplr(copy[ni, :, (EDGE_SIZE + 1):(2 * EDGE_SIZE + 1)])

        # right edge
        copy[ni, :, -EDGE_SIZE:] = np.fliplr(copy[ni, :, (-2 * EDGE_SIZE - 1):(-EDGE_SIZE - 1)])

        # top edge (fills in corners)
        copy[ni, 0:EDGE_SIZE, :] = np.flipud(copy[ni, (EDGE_SIZE + 1):(2 * EDGE_SIZE + 1), :])

        # bottom edge (fills in corners)
        copy[ni, -EDGE_SIZE:, :] = np.flipud(copy[ni, (-2 * EDGE_SIZE - 1):(-EDGE_SIZE - 1), :])

    if DEBUG:
        Image.fromarray(images[0, ...]).save("./debug/1.jpeg")
        Image.fromarray(copy[0, ...]).save("./debug/mirrored_1.jpeg")

    return copy


def pixelToDB(dbFile, images, mirroredImages, labels):
    """
    one pixel becomes one image
    It needs 33 GB space, so directly write to DB
    """

    lmdb_env = lmdb.open(dbFile)

    n, h, w = images.shape
    imageSize = h * w
    for ni in range(n):
        lmdb_txn = lmdb_env.begin(write=True)
        for hi in range(h):
            for wi in range(w):
                i = ni * imageSize + hi * h + wi
                label = labels[ni, hi, wi]
                image = mirroredImages[ni, hi:hi + TILE_SIZE, wi:wi + TILE_SIZE]
                # data is C x H x W array, so add channel axis
                image = image[np.newaxis, ...]

                datum = caffe.io.array_to_datum(image, label)
                lmdb_txn.put(str(id), datum.SerializeToString())

                if i % 10000 == 0:
                    print("\tConverting #%s" % str(i))
                    if DEBUG:
                        Image.fromarray(image[0, ...]).save("./debug/tile_%s.jpeg" % i)

        lmdb_txn.commit()


def convert():
    labels = convertLabels(loadImages(TRAIN_LABELS))

    images = loadImages(TRAIN_IMAGES)
    mirroredImages = mirrorEdges(images)

    trainImages = images[TRAIN_RANGE, ...]
    trainLabel = labels[TRAIN_RANGE, ...]
    mirroredTrainImages = mirroredImages[TRAIN_RANGE, ...]

    testImages = images[TEST_RANGE, ...]
    testLabel = labels[TEST_RANGE, ...]
    mirroredTestImages = mirroredImages[TEST_RANGE, ...]

    print("Start Convert Train Set.")
    pixelToDB(TRAIN_DB, trainImages, mirroredTrainImages, trainLabel)
    print("Start Convert Test Set.")
    pixelToDB(TEST_DB, testImages, mirroredTestImages, testLabel)


if __name__ == "__main__":

    if "--debug" in sys.argv:
        DEBUG = True
        if not os.path.exists("./debug"):
            os.mkdir("./debug")

    print("Start Convert. In Debug Mode: %s" % str(DEBUG))

    convert()
