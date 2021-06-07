def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_display_fn(model):
    display_string = ""
    for name, param in model.named_parameters():
        if param.requires_grad:
            display_string += "Name: {}; Size: {}\n".format(name, param.size())
    display_string += "\n"
    return display_string
