import importlib


def load_hyperconfig(args):
    config_module = importlib.import_module("hyperconfigs.{}_hyperconfigs".format(args.dataset))
    config = getattr(config_module, "{}_hyperconfig".format(args.model))
    config_obj = config()
    config_dict = {}
    obj_attributes = [attribute for attribute in dir(config_obj) if not attribute.startswith('__')]
    for attribute in obj_attributes:
        if attribute != "process_config":
            config_dict[attribute] = eval("config_obj.{}".format(attribute))
    return config_dict, config_obj.process_config
