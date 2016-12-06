import subprocess

import Config


def train(config):
    print ("Start train with " + config.solver)
    args = ["caffe", "train", "--solver=" + config.solver]
    if config.gpu is None:
        args.append("--gpu=all")
    else:
        args.append("--gpu=" + config.gpu)

    p = subprocess.Popen(args, stdout=config.logStream, stderr=config.logStream)

    p.wait()
    config.logStream.flush()


if __name__ == "__main__":
    config = Config.load()
    train(config)
    config.showRunTime()
