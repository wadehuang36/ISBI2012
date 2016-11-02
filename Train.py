import subprocess

import Config


def train(config):
    print ("Start train with " + config.solver)
    p = subprocess.Popen(["caffe", "train", "--solver=" + config.solver],
                         stdout=config.logStream,
                         stderr=config.logStream)
    p.wait()
    config.logStream.flush()


if __name__ == "__main__":
    config = Config.load()
    train(config)
    config.showRunTime()