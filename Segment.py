"""
This segmentation is written by others, so the source code ins't updated
This segmentation has two parts mainly,
 1. Random Forest Training, the applying images are settled on config.randomForestRange
 2. Final Segment, the applying images are settled on config.segmentRange
"""
import subprocess, glob
import numpy as np
import Config
import sklearn.ensemble as ske


def segment(config):
    print ("Start Segment")

    convertLikelihoodNpyToMha(config)

    pmFiles = glob.glob(config.getResultFile("pm_*.mha"))
    pmFiles.sort()
    rawFiles = []
    trustFiles = []
    initSegFiles = []
    treeFiles = []
    saliencyFiles = []
    bclabelFiles = []
    bcfeatFiles = []

    tree2Files = []
    initSeg2Files = []
    bcfeat2Files = []
    bcpredFiles = []
    finalFiles = []

    for i, j in enumerate(config.deployRange):
        print ("\tRunning Image %s" % j)
        rawFiles.append("data/raw/raw_%03d.mha" % j)
        trustFiles.append("data/truth/truth_%03d.png" % j)
        saliencyFiles.append(config.getResultFile("saliency_%03d.ssv" % j))
        initSegFiles.append(config.getResultFile("initseg_%03d.mha" % j))
        treeFiles.append(config.getResultFile("tree_%03d.ssv" % j))
        bcfeatFiles.append(config.getResultFile("bcfeat_%03d.ssv" % j))

        if j in config.randomForestRange:
            bclabelFiles.append(config.getResultFile("bclabel_%03d.ssv" % j))

        if j in config.segmentRange:
            tree2Files.append(config.getResultFile("tree_%03d.ssv" % j))
            initSeg2Files.append(config.getResultFile("initseg_%03d.mha" % j))
            bcfeat2Files.append(config.getResultFile("bcfeat_%03d.ssv" % j))
            bcpredFiles.append(config.getResultFile("bcpred_%03d.ssv" % j))
            finalFiles.append(config.getResultFile("final_%03d.mha" % j))

        print ("\t\tRunning Step 2")
        subprocess.check_call(["hnsWatershed", pmFiles[i], "0.1", "0", "1", "1", initSegFiles[i]])

        print ("\t\tRunning Step 3")
        subprocess.check_call(["hnsMerge", initSegFiles[i], pmFiles[i], "50", "200", "0.5", "0", "1", initSegFiles[i]])

        print ("\t\tRunning Step 4")
        subprocess.check_call(["hnsGenMerges", initSegFiles[i], pmFiles[i], treeFiles[i], saliencyFiles[i]])

        print ("\t\tRunning Step 5")
        subprocess.check_call(
            ["hnsGenBoundaryFeatures", initSegFiles[i], treeFiles[i], saliencyFiles[i], rawFiles[i], pmFiles[i],
             "data/tdict.ssv", bcfeatFiles[i]])

        if j in config.randomForestRange:
            print ("\t\tRunning Step 6")
            subprocess.check_call(
                ["hnsGenBoundaryLabels", initSegFiles[i], treeFiles[i], trustFiles[i], bclabelFiles[i]])

    print ("\tRunning Step 7 And 8")
    x = readSSVs(bcfeatFiles[0:len(bclabelFiles)])
    y = readSSVs(bclabelFiles)

    y = y.reshape(y.size)
    y = y - y.min()
    y = y / y.max() + 1

    rfc = ske.RandomForestClassifier(n_estimators=255, min_samples_split=10)
    rfc.fit(x, y)

    print ("\tRunning Step 9")
    for i, j in enumerate(config.segmentRange):
        x = readSSVs(bcfeat2Files[i:i + 1])  # it needs to be array, so uses []
        x_hat = rfc.apply(x).astype("float32")
        x_hat = x_hat / x_hat.max()
        writeSSV(x_hat, bcpredFiles[i])

        print ("\t\tFinish Segment %s" % j)
        subprocess.check_call(
            ["hnsSegment", initSeg2Files[i], tree2Files[i], bcpredFiles[i], "1", "0", finalFiles[i]])


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
