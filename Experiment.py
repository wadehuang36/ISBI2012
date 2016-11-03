"""
this file is just for test something, it might be broken.
"""
import caffe
import numpy as np
import numpy.lib.format as ft
import matplotlib.pyplot as plt
import struct
from matplotlib.backend_bases import NavigationToolbar2
import sklearn.ensemble as ske
import tifffile

TILE_SIZE = 65
EDGE_SIZE = int((TILE_SIZE - 1) / 2)
TEST_IMAGES = "./data/train-volume.tif"
TEST_LABELS = "./data/train-labels.tif"


def test():
    configFile = "/home/wade/Projects/ISBI2012/models/1/deploy.prototxt"
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
    classifier = caffe.Net("/home/wade/Projects/ISBI2012/models/C/train_test.prototxt", caffe.TRAIN)

    classifier.forward()
    classifier.forward()
    classifier.forward()
    classifier.forward()


curr_pos = 0


def show_likelihood():
    arr = np.load("models/1/likelihood.npy")
    images = arr.reshape(1, 512, 512, 2)

    def handle_back(self, *args, **kwargs):
        global curr_pos
        curr_pos = (curr_pos - 1) % images.shape[0]
        show()

    def handle_forward(self, *args, **kwargs):
        global curr_pos
        curr_pos = (curr_pos + 1) % images.shape[0]
        show()

    NavigationToolbar2.back = handle_back
    NavigationToolbar2.forward = handle_forward

    def show():
        ax.cla()
        ax.imshow(images[curr_pos, :, :, 1], cmap='Greys_r')
        fig.canvas.draw()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    show()

    plt.show()


def show_segment():
    image = np.load("models/A/results/segment_0.npy")
    plt.imshow(image)
    plt.show()


def to_mha():
    arr = np.load("models/A/likelihood.npy")
    imageSize = 512 * 512
    images = arr.reshape(arr.size / imageSize / 2, imageSize, 2)
    images = images.astype("float32")
    for i in range(images.shape[0]):
        with open("models/A/likelihood_%03d.mha" % i, "wb") as mha:
            mha.write("""ObjectType = Image
NDims = 2
BinaryData = True
BinaryDataByteOrderMSB = False
DimSize = 512 512
ElementType = MET_FLOAT
ElementDataFile = LOCAL
""")
            for j in range(images.shape[1]):
                mha.write(images[i, j, 1])

            mha.flush()


def show_mha(fileName):
    arr = np.zeros(512 * 512, dtype="float32")

    with open(fileName, mode="rb") as mha:
        data = mha.read()
        index = data.index("ElementDataFile = LOCAL\n")
        data = data[index + len("ElementDataFile = LOCAL\n"):]
        for i in range(0, len(data), 4):
            arr[i / 4] = struct.unpack("f", data[i:i + 4])[0]

    arr = arr.reshape(512, 512)
    plt.imshow(arr, cmap='Greys_r')
    plt.show()


def rf(Xfiles, Yfiles, Tfiles):
    X = readSSVs(Xfiles)
    Y = readSSVs(Yfiles)

    Y = Y.reshape(Y.size)
    Y = Y - Y.min()
    Y = Y / Y.max() + 1

    rfc = ske.RandomForestClassifier(n_estimators=255, min_samples_split=10)
    rfc.fit(X, Y)

    T = readSSVs(Tfiles)
    T = rfc.apply(T).astype("float32")
    T = T / T.max()
    writeSSV(T,
             "/home/wade/Projects/SegmentationCode/EMSegLiu/jnm14/n3/Result0113-ISBI12-20Training/prediction/bcpred020-1.ssv")


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


def arrayToTif():
    arr = np.load("models/A/results/likelihood.npy")
    arr[arr > 0.5] = 255
    arr[arr <= 0.5] = 0
    arr = arr.astype(np.uint8)

    tifffile.imsave("data/result.tif", arr[:, :, :, 1])

if __name__ == "__main__":
    # test()
    train()
    # show_likelihood()
    # show_segment()
    # to_mha()
    # show_mha("/home/wade/Projects/SegmentationCode/EMSegLiu/jnm14/n3/r20.mha")
    # show_mha("/home/wade/Projects/SegmentationCode/EMSegLiu/jnm14/n3/r20-1.mha")
    # rf(["/home/wade/Projects/SegmentationCode/EMSegLiu/jnm14/n3/Result0113-ISBI12-20Training/feature/bcfeat%03d.ssv" % i for i in range(20)],
    #    ["/home/wade/Projects/SegmentationCode/EMSegLiu/jnm14/n3/Result0113-ISBI12-20Training/label/bclabel%03d.ssv" % i for i in range(20)],
    #    ["/home/wade/Projects/SegmentationCode/EMSegLiu/jnm14/n3/Result0113-ISBI12-20Training/feature/bcfeat020.ssv"])
    # arrayToTif()
