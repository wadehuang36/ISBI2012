"""
This segmentation is written by others, so the source code ins't updated
This segmentation has two parts mainly,
 1. Random Forest Training, the applying images are settled on config.randomForestRange
 2. Final Segment, the applying images are settled on config.segmentRange
"""
import subprocess
import numpy as np
import os
import Config
import sklearn.ensemble as ske
from multiprocessing import Pool


def parallelFunction(args):
    try:
        i = args[0]
        resultsPath = args[1]
        doStep6 = args[2]

        rawFile = "data/raw/raw_%03d.mha" % i
        trustFile = "data/truth/truth_%03d.png" % i
        pmFile = os.path.join(resultsPath, ("pm_%03d.mha" % i))
        initFile = os.path.join(resultsPath, ("initseg_%03d.mha" % i))
        treeFile = os.path.join(resultsPath, ("tree_%03d.ssv" % i))
        saliencyFile = os.path.join(resultsPath, ("saliency_%03d.ssv" % i))
        bcfeatFile = os.path.join(resultsPath, ("bcfeat_%03d.ssv" % i))
        bclabelFile = os.path.join(resultsPath, ("bclabel_%03d.ssv" % i))

        print ("\t\tRunning Image %s Step 2" % i)
        subprocess.check_call(["hnsWatershed", pmFile, "0.1", "0", "1", "1", initFile])

        print ("\t\tRunning Image %s Step 3" % i)
        subprocess.check_call(["hnsMerge", initFile, pmFile, "50", "200", "0.5", "0", "1", initFile])

        print ("\t\tRunning Image %s Step 4" % i)
        subprocess.check_call(["hnsGenMerges", initFile, pmFile, treeFile, saliencyFile])

        print ("\t\tRunning Image %s Step 5" % i)
        subprocess.check_call(
            ["hnsGenBoundaryFeatures", initFile, treeFile, saliencyFile, rawFile, pmFile, "data/tdict.ssv", bcfeatFile])

        if doStep6:
            print ("\t\tRunning Image %s Step 6" % i)
            subprocess.check_call(["hnsGenBoundaryLabels", initFile, treeFile, trustFile, bclabelFile])

        return True
    except Exception, e:
        print e
        return False


def segment(config):
    print ("Start Segment")
    convertLikelihoodNpyToMha(config)

    # Can't pass Config object, so pass every thing needs
    p = Pool()
    results = p.map(parallelFunction,
                    [(i, config.resultsPath, i in config.randomForestRange) for i in config.deployRange])

    if len(filter(lambda x: x == False, results)) > 0:
        print "Segment Fails"
        return

    print ("\tRunning Step 7 And 8")
    x = readSSVs([config.getResultFile("bcfeat_%03d.ssv" % i) for i in config.randomForestRange])
    y = readSSVs([config.getResultFile("bclabel_%03d.ssv" % i) for i in config.randomForestRange])

    y = y.reshape(y.size)
    y = y - y.min()
    y = y / y.max() + 1

    rfc = ske.RandomForestClassifier(n_estimators=255, min_samples_split=10)
    rfc.fit(x, y)

    print ("\tRunning Step 9")
    for i in config.segmentRange:
        x = readSSVs([config.getResultFile("bcfeat_%03d.ssv" % i)])  # it needs to be array, so uses []
        x_hat = rfc.apply(x).astype("float32")
        x_hat = x_hat / x_hat.max()
        writeSSV(x_hat, config.getResultFile("bcpred_%03d.ssv" % i))

        print ("\t\tFinish Segment %s" % i)
        subprocess.check_call(
            ["hnsSegment",
             config.getResultFile("initseg_%03d.mha" % i),
             config.getResultFile("tree_%03d.ssv" % i),
             config.getResultFile("bcpred_%03d.ssv" % i),
             "1",
             "0",
             config.getResultFile("final_%03d.mha" % i)])


def convertLikelihoodNpyToMha(config):
    arr = np.load(config.likelihood)
    # become number, height* width, channel
    images = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2], 2)
    images = images.astype("float32")

    for i, j in enumerate(config.deployRange):
        with open(config.getResultFile("pm_%03d.mha" % j), "wb") as mha:
            mha.write("""ObjectType = Image
NDims = 2
BinaryData = True
BinaryDataByteOrderMSB = False
DimSize = 512 512
ElementType = MET_FLOAT
ElementDataFile = LOCAL
""")
            for k in range(images.shape[1]):
                mha.write(images[i, k, 0])

            mha.flush()


def readSSVs(files):
    R = []
    for name in files:
        with open(name, "r") as f:
            for line in f:
                s = [float(i) for i in line.split(" ")]
                R.append(s)

    return np.array(R)


def writeSSV(R, fileName):
    with open(fileName, "w") as f:
        for i in range(R.shape[0]):
            f.write(str(R[i, 0]) + "\n")


if __name__ == "__main__":
    config = Config.load()
    segment(config)
    config.showRunTime()
