import Config
import Convert
import Train
import Deploy
import Segment2

if __name__ == "__main__":
    config = Config.load()
    Convert.convert(config)
    Train.train(config)
    Deploy.deploy(config)
    Segment2.segment(config)
