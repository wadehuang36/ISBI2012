import Config
import Convert
import Train
import Deploy

if __name__ == "__main__":
    config = Config.load()
    Convert.convert(config)
    Train.train(config)
    Deploy.deploy(config)
