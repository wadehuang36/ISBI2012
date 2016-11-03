import subprocess

import Config
import numpy as np
import glob
import tifffile
import Convert
import sys


def evaluation_likelihood(config):
    print ("Evaluating Likelihood")
    arr = np.load(config.likelihood)
    arr[arr > 0.5] = 255
    arr[arr <= 0.5] = 0
    arr = arr.astype(np.uint8)

    likelihood = config.getResultFile("likelihood.tif")
    tifffile.imsave(likelihood, arr[:, :, :, 1])

    p = subprocess.Popen(["java", "-jar", "Evaluation.jar", config.trainLabels, likelihood],
                         stdout=config.logStream,
                         stderr=config.logStream)
    p.wait()
    config.logStream.flush()


def evaluation_final(config):
    print ("Evaluating Final Result")
    arr = Convert.loadMHA(glob.glob(config.getResultFile("final_*.mha")))
    arr[arr != 0.0] = 255
    arr = arr.astype(np.uint8)

    final = config.getResultFile("final.tif")
    tifffile.imsave(final, arr[:, :, :])

    p = subprocess.Popen(["java", "-jar", "Evaluation.jar", config.trainLabels, final],
                         stdout=config.logStream,
                         stderr=config.logStream)
    p.wait()
    config.logStream.flush()


if __name__ == "__main__":
    config = Config.load()
    if "-nl" not in sys.argv:
        evaluation_likelihood(config)

    if "-nf" not in sys.argv:
        evaluation_final(config)
