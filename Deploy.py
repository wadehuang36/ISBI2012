import caffe
import numpy as np
import Config
import Convert


def deploy(config):
    caffe.set_mode_gpu()
    if config.gpu is not None:
        caffe.set_device(int(config.gpu))

    classifier = caffe.Classifier(config.modelPrototxt, config.trainedModel)
    images = Convert.loadImages(config.deployImages)
    if config.deployRange is not None:
        images = images[config.deployRange, ...]

    mirroredImages = Convert.mirrorEdges(config.subImageSize, images, config.debug) * 0.00390625
    probs = np.zeros((images.size, 2))
    n, h, w = images.shape
    b = config.deployBatch
    imageSize = h * w

    print("Start Deploying")
    for ni in range(n):
        print("\tDeploying Image %s" % config.deployRange[ni])
        for hi in range(h):
            for wi in range(0, w, b):
                batches = []
                for bi in range(wi, wi + b):
                    image = mirroredImages[ni, hi:hi + config.subImageSize, bi:bi + config.subImageSize]
                    # data is K x H x W X C array, so add channel axis
                    batches.append(image[:, :, np.newaxis])

                prob = classifier.predict(batches, oversample=False)
                i = ni * imageSize + hi * h + wi
                probs[i:i+b] = prob

    probs = probs.reshape(n, h, w, 2)
    np.save(config.likelihood, probs)


if __name__ == "__main__":
    config = Config.load()
    deploy(config)
    config.showRunTime()
