"""
this file is just for test something, it might be broken.
"""
import caffe
import numpy as np
import matplotlib.pyplot as plt

TILE_SIZE = 65
EDGE_SIZE = int((TILE_SIZE - 1) / 2)
TEST_IMAGES = "./data/train-volume.tif"
TEST_LABELS = "./data/train-labels.tif"


def test():
    configFile = "/home/wade/Projects/ISBI2012/models/1/full.prototxt"
    trainedModel = "/home/wade/Projects/ISBI2012/snapshot/1/full_iter_4000.caffemodel.h5"
    classifier = caffe.Net(configFile, caffe.TEST, weights=trainedModel)

    convert = __import__("1_convert")
    images = convert.loadImages(TEST_IMAGES)
    mirroredImages = convert.mirrorEdges(images)

    labels = convert.convertLabels(convert.loadImages(TEST_LABELS))

    results = np.zeros((images.size, 2))
    n, h, w = images.shape
    imageSize = h * w

    correctCount = 0
    for ni in range(0, 1):
        for hi in range(h):
            for wi in range(w):
                i = ni * imageSize + hi * h + wi
                image = mirroredImages[ni, hi:hi + TILE_SIZE, wi:wi + TILE_SIZE]
                # data is K x H x W X C array, so add channel axis
                image = image[np.newaxis, np.newaxis, :, :]
                classifier.blobs["data"].data[...] = image * 0.00390625
                out = classifier.forward()
                result = out["prob"][0]
                results[i, ...] = result

                label = result.argmax()
                trueLabel = labels[ni, hi, wi]

                if label == trueLabel:
                    correctCount += 1

                if i % 100 == 0:
                    print("\tApply #%s, Accuracy: %s" % (i, correctCount / (i + 1.0)))

    results = results.reshape(results.shapes[0], results.shapes[1], results.shapes[2], 2)
    np.save("test.npy", results)


def train():
    classifier = caffe.Net("/home/wade/Projects/ISBI2012/models/1/full_train_test.prototxt", caffe.TRAIN)

    for ni in range(0, 512 * 512 / 64):
        out = classifier.forward()
        print out


def show():
    arr = np.load("test.npy")
    images = arr.reshape(30, 512, 512, 2)
    image = images[0, :, :, 1]
    plt.imshow(image, cmap='Greys_r')
    plt.show()
    plt.show()


if __name__ == "__main__":
    # test()
    # train()
    show()
