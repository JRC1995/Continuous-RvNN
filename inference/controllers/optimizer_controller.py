from optimizers import *
import torch as T

def get_optimizer(config):
    if config["optimizer"].lower() == "ranger":
        return Ranger
    elif config["optimizer"].lower() == "adam" or config["optimizer"].lower() == "adam_":
        return T.optim.Adam
    elif config["optimizer"].lower() == "adagrad":
        return T.optim.Adagrad
    elif config["optimizer"].lower() == "adadelta":
        return T.optim.Adadelta
    else:
        return RAdam
