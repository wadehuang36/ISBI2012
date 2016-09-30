import Config
import convert
import train
import deploy

if __name__ == "__main__":
    config = Config.load()
    convert.convert(config)
    train.train(config)
    deploy.deploy(config)
