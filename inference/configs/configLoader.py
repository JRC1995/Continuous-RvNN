import importlib


def load_config(args):

    config_module = importlib.import_module("configs.{}_configs".format(args.dataset))
    config = getattr(config_module, "{}_config".format(args.model))
    config_obj = config()
    config_dict = {}
    obj_attributes = [attribute for attribute in dir(config_obj) if not attribute.startswith('__')]
    for attribute in obj_attributes:
        config_dict[attribute] = eval("config_obj.{}".format(attribute))

    if "penalty_gamma" in config_dict and args.penalty_gamma != -1:
        config_dict["penalty_gamma"] = args.penalty_gamma

    config_dict["dataset"] = args.dataset

    return config_dict
