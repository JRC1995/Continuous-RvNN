import random


def step_display_fn(epoch, iter, item, args, config, extras):
    if args.model_type == "NLI" or args.model_type == "Classifier":
        display_string = "Model: {}, Dataset: {}, Current LR: {}, Epoch {}, Step: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(
            config["model_name"],
            args.dataset,
            config["current_lr"],
            epoch,
            iter,
            item["metrics"]["loss"],
            item["metrics"]["accuracy"])
    return display_string


def example_display_fn(epoch, iter, item, args, config, extras):
    if args.model_type == "NLI" or args.model_type == "Classifier":
        idx2labels = extras["idx2labels"]
        item_len = len(item["display_items"]["predictions"])
        chosen_id = random.choice([id for id in range(item_len)])

        if args.model_type == "NLI":
            display_string = "Example:\nSequence1: {}\nSequence2: {}\nPrediction: {}\nGround Truth: {}\n".format(
                " ".join(item["display_items"]["sequences1"][chosen_id]),
                " ".join(item["display_items"]["sequences2"][chosen_id]),
                idx2labels[item["display_items"]["predictions"][chosen_id]],
                idx2labels[item["display_items"]["labels"][chosen_id]])
        else:
            display_string = "Example:\nSequence1: {}\nPrediction: {}\nGround Truth: {}\n".format(
                " ".join(item["display_items"]["sequences"][chosen_id]),
                idx2labels[item["display_items"]["predictions"][chosen_id]],
                idx2labels[item["display_items"]["labels"][chosen_id]])


    return display_string

def display(display_string, log_paths):
    with open(log_paths["log_path"], "a") as fp:
        fp.write(display_string)
    with open(log_paths["verbose_log_path"], "a") as fp:
        fp.write(display_string)
    print(display_string)

