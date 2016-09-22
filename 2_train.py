import datetime, subprocess, os, sys
import glob
import argparse

# Path to caffe
CAFFE_PATH = "/home/wade/caffe/bin"


def train(config):
    # commands.getoutput("cat %s" % config)
    caffe = os.path.join(CAFFE_PATH, "caffe")
    subprocess.call([caffe, "train", "--solver=" + config])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sovler", dest="solverFile")

    args = parser.parse_args()

    if args.solverFile is None:
        print ("Place choose a solver file")
        files = glob.glob("configs/*/*_solver.prototxt")
        for i in range(len(files)):
            print("%s ) %s" % (i + 1, files[i]))

        chose = raw_input("Enter the number: ")
        solverFile = files[int(chose) - 1]
    else:
        solverFile = args.solverFile

    print ("Start train with " + solverFile)
    train(solverFile)
