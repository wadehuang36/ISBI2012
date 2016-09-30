import caffe
import numpy as np
import Config
import convert
import os

TILE_SIZE = 65
EDGE_SIZE = int((TILE_SIZE - 1) / 2)
TEST_IMAGES = "./data/test-volume.tif"


def deploy(config):
    classifier = caffe.Classifier(config.modelPrototxt, config.trainedModel)

    images = convert.loadImages(config.deployImages)
    if config.deployRange is not None:
        images = images[config.deployRange, ...]

    mirroredImages = convert.mirrorEdges(images)
    probs = np.zeros((images.size, 2))
    n, h, w = images.shape
    imageSize = h * w

    print("Start Deploying")
    for ni in range(0, n):
        for hi in range(h):
            for wi in range(w):
                i = ni * imageSize + hi * h + wi
                image = mirroredImages[ni, hi:hi + TILE_SIZE, wi:wi + TILE_SIZE]
                # data is K x H x W X C array, so add channel axis
                image = image[np.newaxis, :, :, np.newaxis]
                prob = classifier.predict(image)[0]
                probs[i, ...] = prob

                if i % 10000 == 0:
                    print("\tDeployed #%s" % str(i))

    np.save(config.likelihood, probs)


if __name__ == "__main__":
    deploy(Config.load())
