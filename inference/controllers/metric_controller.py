
def metric_fn(metrics, args):
    if args.model_type == "NLI" or args.model_type == "Classifier":

        correct_predictions = sum([metric["correct_predictions"] for metric in metrics])
        total = sum([metric["total"] for metric in metrics])
        accuracy = correct_predictions/total if total > 0 else 0
        loss = sum([metric["loss"] for metric in metrics])/len(metrics) if len(metrics) > 0 else 0

        composed_metric = {"loss": loss,
                           "accuracy": accuracy*100}

    return composed_metric


def compose_dev_metric(metrics, args, config):
    if args.model_type == "Classifier" or args.model_type == "NLI":
        total_metric = 0
        n = len(metrics)
        for key in metrics:
            total_metric += metrics[key][config["save_by"]]
        return config["metric_direction"] * total_metric / n
