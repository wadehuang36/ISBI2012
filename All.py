import Config
import Convert
import Train
import Deploy
import Segment
import Evaluation

if __name__ == "__main__":
    config = Config.load()

    Convert.convert(config)

    Train.train(config)
    config.showRunTime()

    Deploy.deploy(config)
    config.showRunTime()

    Segment.segment(config)
    Evaluation.evaluation(config)
    config.showRunTime()