import caffe
import Config
import Convert
import numpy as np


def train(config):
    solver = caffe.get_solver(config.solver)

    labels = Convert.convertLabels(Convert.loadImages(config.trainLabels)).astype(np.float32)
    images = Convert.loadImages(config.deployImages).astype(np.float32)
    if config.trainRange is not None:
        labels = labels[config.deployRange, ...]
        images = images[config.deployRange, ...]

    n, h, w = images.shape
    images = Convert.mirrorEdges(config.subImageSize, images, config.debug) * 0.00390625
    b = config.trainBatch

    print("Start Training")
    for ni in range(n):
        print("\tTraining Image %s" % ni)
        for hi in range(h):
            for wi in range(0, w, b):
                trainX = np.zeros((b, 1, config.subImageSize, config.subImageSize), np.float32)
                for bi in range(wi, wi + b):
                    image = images[ni, hi:hi + config.subImageSize, bi:bi + config.subImageSize]
                    # data is K x C X H x W array, so add channel axis
                    trainX[bi - wi] = image[np.newaxis, :, :]

                trainY = labels[ni, hi, wi: wi + b]
                solver.net.blobs["data"].data[...] = trainX
                solver.net.blobs["label"].data[...] = trainY
                solver.step(b)

        solver.net.save(config.trainedModel)


if __name__ == "__main__":
    config = Config.load()
    train(config)
    config.showRunTime()
