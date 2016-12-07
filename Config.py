import json
import sys
import os
import argparse
import glob
import logging
import datetime
import time


class Config:
    def __init__(self, modelFile, args):
        config = json.load(open(modelFile))

        self.gpu = args.gpu
        self.debug = args.debug
        self.no_convert = args.no_convert
        self.no_train = args.no_train
        self.no_deploy = args.no_deploy
        self.no_segment = args.no_segment
        self.no_eval = args.no_eval

        self.modelPath = os.path.dirname(modelFile)
        self.subImageSize = int(config["subImageSize"])
        self.trainData = config["trainData"]
        self.trainImages = config["trainImages"]
        self.trainLabels = config["trainLabels"]
        self.trainRange = eval(config["trainRange"])

        if "trainBatch" in config:
            self.trainBatch = int(config["trainBatch"])
        else:
            self.trainBatch = 128

        self.testData = config["testData"]
        self.testImages = config["testImages"]
        self.testLabels = config["testLabels"]

        if config["testRange"] is not None:
            self.testRange = eval(config["testRange"])

        self.deployImages = config["deployImages"]
        self.deployLabels = config["deployLabels"]
        self.deployRange = eval(config["deployRange"])

        if "deployBatch" in config:
            self.deployBatch = int(config["deployBatch"])
        else:
            self.deployBatch = 128

        self.randomForestRange = eval(config["randomForestRange"])
        self.segmentRange = eval(config["segmentRange"])

        self.solver = str(config["solver"])
        self.modelPrototxt = str(config["modelPrototxt"])

        if args.trainedModel is None:
            self.trainedModel = str(config["trainedModel"])
        else:
            self.trainedModel = str(args.trainedModel)

        self.likelihood = config["likelihood"]
        self.segment = config["segment"]

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.resultsPath = os.path.join(self.modelPath, "results")
        if not os.path.exists(self.resultsPath):
            os.mkdir(self.resultsPath)

        # make out to file and console
        if args.log:
            logFile = os.path.join(self.resultsPath, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M.log"))
            fh = logging.FileHandler(logFile)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.stdout = sys.stdout
            self.stderr = sys.stderr
            self.logStream = fh.stream
            self.logger.addHandler(fh)
            self.logger.addHandler(logging.StreamHandler())
            sys.stdout = StdWrapper(self.logger, logging.DEBUG)
            sys.stderr = StdWrapper(self.logger, logging.ERROR)
        else:
            self.logStream = sys.stdout

        self.start = time.time()

    def getResultFile(self, fileName):
        return os.path.join(self.resultsPath, fileName)

    def showRunTime(self):
        intervals = (
            ('weeks', 604800),
            ('days', 86400),
            ('hours', 3600),
            ('mins', 60),
            ('secs', 1),
        )

        seconds = time.time() - self.start
        result = []
        for name, count in intervals:
            value = int(seconds // count)
            if value:
                seconds -= value * count
                if value == 1:
                    name = name.rstrip('s')
                result.append("{} {}".format(value, name))
        print ', '.join(result)


class StdWrapper:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, msg):
        msg = msg.strip()
        if msg:
            self.logger.log(self.level, msg)

    def flush(self):
        pass


instance = None


def load(log=True):
    global instance
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model")
    parser.add_argument("--trainedModel", dest="trainedModel", default=None)
    parser.add_argument("--gpu", dest="gpu", default=None)
    parser.add_argument("--nc", dest="no_convert", const=True, action='store_const', default=False)
    parser.add_argument("--nt", dest="no_train", const=True, action='store_const', default=False)
    parser.add_argument("--nd", dest="no_deploy", const=True, action='store_const', default=False)
    parser.add_argument("--ns", dest="no_segment", const=True, action='store_const', default=False)
    parser.add_argument("--ne", dest="no_eval", const=True, action='store_const', default=False)
    parser.add_argument("--debug", dest="debug", const=True, action='store_const', default=False)
    parser.add_argument("--nolog", dest="nolog", const=True, action='store_const', default=False)
    args, unknownArgs = parser.parse_known_args()

    if args.model is None:
        print ("Place choose a model file")
        files = glob.glob("models/*/config.json")
        files.sort()
        for i in range(len(files)):
            print("%s ) %s" % (i + 1, files[i]))

        if sys.version_info.major == 3:
            chose = input("Enter the number: ")
        else:
            chose = raw_input("Enter the number: ")

        modelFile = files[int(chose) - 1]
    else:
        modelFile = "models/%s/config.json" % args.model

    args.log = log and not args.nolog
    instance = config = Config(modelFile, args)

    if args.debug:
        if not os.path.exists("./debug"):
            os.mkdir("./debug")

    return config
