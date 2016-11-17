import numpy as np
import os
import SimpleITK.SimpleITK as itk
import Config
import Convert
import scipy.ndimage as ndimage
import cv2


def segment(config):
    # images = Convert.loadImages(config.trainImages)
    # gts = Convert.convertLabels(Convert.loadImages(config.trainLabels))
    # probs = np.load(config.likelihood) * 256

    images = Convert.loadImages(config.trainImages)
    gts = _getGroundTrues(Convert.loadImages(config.trainLabels))
    probs = Convert.loadImages(config.likelihood)

    RandIndex = np.zeros(probs.shape[0])
    VOI = np.zeros(probs.shape[0])
    RE = np.zeros(probs.shape[0])

    for i in range(probs.shape[0]):
        pr = probs[i]
        temp = np.zeros((pr.shape[0] + 2, pr.shape[1] + 2))
        temp[1:pr.shape[0] + 1, 1:pr.shape[1] + 1] = pr[:, :]
        pr = temp
        pr = ndimage.gaussian_filter(pr, sigma=5, mode="constant")
        thres = 0.03
        itk.Water

            # likelihoodImages = []
            # for i in range(probs.shape[0]):
            #     likelihoodImages.append(itk.GetImageFromArray(probs[i, :, :, 1]))

            # output = _watershed(likelihoodImages)
            # output = _merge(output)

            # _save(config, probs.shape[0:3], output)


def _watershed(images):
    watershedImages = []
    for i in range(len(images)):
        output = itk.MorphologicalWatershed(images[i], level=0.3, fullyConnected=True, markWatershedLine=True)
        watershedImages.append(output)

    return watershedImages


def _merge(images):
    mergedImages = []
    for i in range(len(images)):
        output = itk.RegionalMaxima(images[i])
        mergedImages.append(output)

    return mergedImages


def _save(config, size, images):
    outputArray = np.zeros(size)

    for i in range(size[0]):
        outputArray[i, ...] = itk.GetArrayFromImage(images[i])
    np.save(config.segment, outputArray)

def _getGroundTrues(images):
    gts = np.zeros(images.shape)

    for i in range(images.shape[0]):
        gt = images[i]
        temp = np.copy(gt)
        uni = np.unique(gt)
        counter = 1
        for u in uni[1:]:
            temp[temp == u] = counter
            counter += 1

        gts[i] = temp
    return gts


if __name__ == "__main__":
    segment(Config.load())
