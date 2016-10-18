import caffe
import numpy as np
import Config
import Convert
import os


def deploy(config):
    classifier = caffe.Classifier(config.modelPrototxt, config.trainedModel)

    images = Convert.loadImages(config.deployImages)
    if config.deployRange is not None:
        images = images[config.deployRange, ...]

    mirroredImages = Convert.mirrorEdges(config.subImageSize, images, config.debug)
    probs = np.zeros((images.size, 2))
    n, h, w = images.shape
    imageSize = h * w

    print("Start Deploying")
    for ni in range(n):
        for hi in range(h):
            for wi in range(w):
                i = ni * imageSize + hi * h + wi
                image = mirroredImages[ni, hi:hi + config.subImageSize, wi:wi + config.subImageSize]
                # data is K x H x W X C array, so add channel axis
                image = image[np.newaxis, :, :, np.newaxis] * 0.00390625
                prob = classifier.predict(image, oversample=False)[0]

                probs[i, ...] = prob

                if i % 1000 == 0:
                    print("\tDeployed #%s" % str(i))

    probs = probs.reshape(n, h, w, 2)
    np.save(config.likelihood, probs)


if __name__ == "__main__":
    deploy(Config.load())
