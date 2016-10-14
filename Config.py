import json
import sys
import os
import argparse
import glob
import logging
import datetime


class Config:
    def __init__(self, modelFile, debug):
        config = json.load(open(modelFile))
        self.modelPath = os.path.dirname(modelFile)
        self.debug = debug
        self.subImageSize = int(config["subImageSize"])

        self.trainData = config["trainData"]
        self.trainImages = config["trainImages"]
        self.trainLabels = config["trainLabels"]
        if "trainRange" in config:
            self.trainRange = eval(config["trainRange"])

        self.testData = config["testData"]
        self.testImages = config["testImages"]
        self.testLabels = config["testLabels"]
        if "testRange" in config:
            self.testRange = eval(config["testRange"])

        self.deployImages = config["deployImages"]
        if "deployRange" in config:
            self.deployRange = eval(config["deployRange"])

        self.solver = config["solver"]
        self.modelPrototxt = str(config["modelPrototxt"])
        self.trainedModel = str(config["trainedModel"])
        self.likelihood = config["likelihood"]
        self.segment = config["segment"]

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.resultsPath = os.path.join(self.modelPath, "results")
        if not os.path.exists(self.resultsPath):
            os.mkdir(self.resultsPath)

        # make out to file and console
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

    def getResultFile(self, fileName):
        return os.path.join(self.resultsPath, fileName)


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


def load():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model")
    parser.add_argument("--debug", dest="debug")
    args, unknown = parser.parse_known_args()

    if args.model is None:
        print ("Place choose a model file")
        files = glob.glob("models/*/config.json")
        for i in range(len(files)):
            print("%s ) %s" % (i + 1, files[i]))

        if sys.version_info.major == 3:
            chose = input("Enter the number: ")
        else:
            chose = raw_input("Enter the number: ")

        modelFile = files[int(chose) - 1]
    else:
        modelFile = "models/%s/config.json" % args.model

    debug = "--debug" in sys.argv
    config = Config(modelFile, debug)

    if debug:
        if not os.path.exists("./debug"):
            os.mkdir("./debug")

    return config
