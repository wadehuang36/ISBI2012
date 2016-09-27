import caffe
import numpy as np
import glob
import argparse

TILE_SIZE = 65
EDGE_SIZE = int((TILE_SIZE - 1) / 2)
TEST_IMAGES = "./data/train-volume.tif"
TEST_LABELS = "./data/train-labels.tif"


def apply(configFile, trainedModel):
    classifier = caffe.Classifier(configFile,
                                  trainedModel)

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
                image = image[np.newaxis, :, :, np.newaxis] * 0.00390625
                result = classifier.predict(image, oversample=False)
                results[i, ...] = result[0]

                trueResult = labels[ni, hi, wi]

                label1 = result[0, 0]
                label2 = result[0, 1]
                if trueResult == 0 and label1 > label2:
                    correctCount += 1
                elif trueResult == 1 and label1 < label2:
                    correctCount += 1
                else:
                    pass

                if i % 100 == 0:
                    print("\tApply #%s, Accuracy: %s" % (i, correctCount / (i + 1)))

    np.save("test.npy", results)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", dest="modelFile")
    #
    # args = parser.parse_args()
    #
    # if args.modelFile is None:
    #     print ("Place choose a model")
    #     files = glob.glob("snapshot/*/*.caffemodel.h5")
    #     for i in range(len(files)):
    #         print("%s ) %s" % (i + 1, files[i]))
    #
    #     chose = raw_input("Enter the number: ")
    #     modelFile = files[int(chose) - 1]
    # else:
    #     modelFile = args.modelFile
    #
    # print ("Start train with " + modelFile)
    configFile = "/home/wade/Projects/ISBI2012/configs/1/full_apply.prototxt"
    trainedModel = "/home/wade/Projects/ISBI2012/snapshot/1/full_iter_379.caffemodel.h5"
    apply(configFile, trainedModel)
