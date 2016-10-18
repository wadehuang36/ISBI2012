import numpy as np
import os
import SimpleITK.SimpleITK as itk
import Config


def segment(config):
    array = np.load(config.likelihood)
    likelihoodImages = []
    for i in range(array.shape[0]):
        likelihoodImages.append(itk.GetImageFromArray(array[i, :, :, 1]))

    output = _watershed(likelihoodImages)
    #output = _merge(output)

    _save(config, array.shape[0:3], output)


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


if __name__ == "__main__":
    segment(Config.load())
