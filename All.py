import Config
import Convert
import Train
import Deploy
import Segment
import Evaluation

if __name__ == "__main__":
    config = Config.load()

    if not config.no_convert:
        Convert.convert(config)
        config.showRunTime()

    if not config.no_train:
        Train.train(config)
        config.showRunTime()

    if not config.no_deploy:
        Deploy.deploy(config)
        config.showRunTime()

    if not config.no_segment:
        Segment.segment(config)
        config.showRunTime()

    if not config.no_eval:
        Evaluation.evaluation(config)

    config.showRunTime()
