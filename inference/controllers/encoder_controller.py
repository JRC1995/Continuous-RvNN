from models.encoders import *

def encoder(config):
    return eval(config["encoder"])(config=config)

