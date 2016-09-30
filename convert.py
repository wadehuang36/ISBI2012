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
import caffe
import numpy as np
import lmdb
import Config
from PIL import Image


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


def mirrorEdges(subImageSize, images, debug):
    """
    Padding edges by mirror to mirror + 512 + mirror
    :param images:
    :return:
    """
    n, h, w = images.shape
    edgeSize = int((subImageSize - 1) / 2)
    copy = np.zeros((n, h + subImageSize - 1, w + subImageSize - 1), dtype=images.dtype)
    for ni in range(n):
        # inside is the same
        copy[ni, edgeSize:h + edgeSize, edgeSize:w + edgeSize] = images[ni, ...]

        # left edge
        copy[ni, :, 0:edgeSize] = np.fliplr(copy[ni, :, (edgeSize + 1):(2 * edgeSize + 1)])

        # right edge
        copy[ni, :, -edgeSize:] = np.fliplr(copy[ni, :, (-2 * edgeSize - 1):(-edgeSize - 1)])

        # top edge (fills in corners)
        copy[ni, 0:edgeSize, :] = np.flipud(copy[ni, (edgeSize + 1):(2 * edgeSize + 1), :])

        # bottom edge (fills in corners)
        copy[ni, -edgeSize:, :] = np.flipud(copy[ni, (-2 * edgeSize - 1):(-edgeSize - 1), :])

    if debug:
        Image.fromarray(images[0, ...]).save("./debug/1.jpeg")
        Image.fromarray(copy[0, ...]).save("./debug/mirrored_1.jpeg")

    return copy


def pixelToDB(dbFile, subImageSize, images, mirroredImages, labels, debug):
    """
    one pixel becomes one image
    It needs 33 GB space, so directly write to DB
    """

    lmdb_env = lmdb.open(dbFile, map_size=int(1e12))

    n, h, w = images.shape
    imageSize = h * w
    for ni in range(n):
        lmdb_txn = lmdb_env.begin(write=True)
        for hi in range(h):
            for wi in range(w):
                i = ni * imageSize + hi * h + wi
                label = labels[ni, hi, wi]
                image = mirroredImages[ni, hi:hi + subImageSize, wi:wi + subImageSize]
                # data is C x H x W array, so add channel axis
                image = image[np.newaxis, ...]

                datum = caffe.io.array_to_datum(image, label)
                keystr = '{:0>8d}'.format(i)
                lmdb_txn.put(keystr, datum.SerializeToString())

                if i % 10000 == 0:
                    print("\tConverting #%s" % str(i))
                    if debug:
                        Image.fromarray(image[0, ...]).save("./debug/tile_%s.jpeg" % i)

        lmdb_txn.commit()


def convert(config):
    if os.path.exists(config.trainData):
        print("%s exists, skip converting" % config.trainData)
    else:
        labels = convertLabels(loadImages(config.trainLabels))

        images = loadImages(config.trainImages)
        mirroredImages = mirrorEdges(config.subImageSize, images, config.debug)

        trainImages = images[config.trainRange, ...]
        trainLabel = labels[config.trainRange, ...]
        mirroredTrainImages = mirroredImages[config.trainRange, ...]

        testImages = images[config.testRange, ...]
        testLabel = labels[config.testRange, ...]
        mirroredTestImages = mirroredImages[config.testRange, ...]

        print("Start Convert Train Set.")
        pixelToDB(config.trainData, config.subImageSize, trainImages, mirroredTrainImages, trainLabel, config.debug)
        print("Start Convert Test Set.")
        pixelToDB(config.testData, config.subImageSize, testImages, mirroredTestImages, testLabel, config.debug)


if __name__ == "__main__":
    convert(Config.load())
