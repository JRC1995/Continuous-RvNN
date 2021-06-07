import random
import zlib
from pathlib import Path

import numpy as np
import torch.nn as nn

from collaters import *
from configs.configLoader import load_config
from controllers.attribute_controller import prepare_attributes
from controllers.extras_controller import extras_fn
from controllers.metric_controller import metric_fn, compose_dev_metric
from parser import get_args
from trainers import Trainer
from utils.checkpoint_utils import load_temp_checkpoint, load_infer_checkpoint, save_infer_checkpoint, \
    save_temp_checkpoint
from utils.data_utils import load_data, load_dataloaders
from utils.display_utils import example_display_fn, step_display_fn, display
from utils.param_utils import param_display_fn, param_count
from utils.path_utils import load_paths
from models import *
from agents import *
import numpy as np

device = T.device('cuda' if T.cuda.is_available() else 'cpu')


def run(args, config, time=0):
    global device

    SEED = "{}_{}_{}_{}".format(args.dataset, args.model, args.model_type, time)
    SEED = zlib.adler32(str.encode(SEED))
    display_string = "\n\nSEED: {}\n\n".format(SEED)
    display_string += "Parsed Arguments: {}\n\n".format(args)

    if args.reproducible:
        T.manual_seed(SEED)
        random.seed(SEED)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        # T.set_deterministic(True)

    display_string += "Configs:\n"
    for k, v in config.items():
        display_string += "{}: {}\n".format(k, v)
    display_string += "\n"

    paths, checkpoint_paths, metadata = load_paths(args, time)

    data = load_data(paths, metadata, args)

    attributes = prepare_attributes(data, args)

    model = eval("{}_model".format(args.model_type))
    model = model(attributes=attributes,
                  config=config)
    model = model.to(device)

    if config["DataParallel"]:
        model = nn.DataParallel(model)

    if args.display_params:
        display_string += param_display_fn(model)

    total_parameters = param_count(model)
    display_string += "Total Parameters: {}\n\n".format(total_parameters)

    print(display_string)

    if not args.checkpoint:
        with open(paths["verbose_log_path"], "w+") as fp:
            fp.write(display_string)
        with open(paths["log_path"], "w+") as fp:
            fp.write(display_string)

    agent = eval("{}_agent".format(args.model_type))

    agent = agent(model=model,
                  config=config,
                  device=device)

    agent, epochs_taken = load_infer_checkpoint(agent, checkpoint_paths, paths)
    idx2vocab = data["idx2vocab"]
    UNK_id = data["UNK_id"]

    texts = ["this is the song which i love the most .",
             "i like this very much .",
             "i did not like a single minute of this film .",
             "roger dodger is one of the most compelling variations of this theme",
             "recursive neural networks can compose sequences according to their underlying hierarchical syntactic structures .",
             "Recursive Neural Networks (RvNNs), which compose sequences according to their underlying hierarchical syntactic structure, have been shown to perform well in several natural language processing tasks compared to similar models without structural biases . "]

    #texts = ["roger dodger is one of the most compelling variation of this theme ."]
    for text in texts:
        text = text.lower().split(" ")
        text_idx = [idx2vocab.get(word, UNK_id) for word in text]
        input_mask = [1] * len(text_idx)

        batch = {}
        batch["sequences1_vec"] = T.tensor([text_idx]).long()
        batch["sequences2_vec"] = T.tensor([text_idx]).long()
        batch["input_masks1"] = T.tensor([input_mask]).float()
        batch["input_masks2"] = T.tensor([input_mask]).float()
        batch["temperature"] = None

        agent.model.module.encoder.composition_scores = []
        output_dict = agent.model(batch)
        comp_scores = agent.model.module.encoder.composition_scores

        #print(comp_scores)

        struct = ["NULL"] * len(text_idx)

        cumulative_comp_score = [0] * len(text_idx)

        S = len(text_idx)

        existence_prob = [1] * S

        existence_prob = np.asarray(existence_prob)
        cumulative_comp_score = np.asarray(cumulative_comp_score)

        step = 0

        def set_closing(i, struct):
            if struct[i + 1] == "NULL":
                struct[i + 1] = ")"
            elif "(" in struct[i + 1]:
                close_count = 0
                open_count = len(struct[i + 1])
                j = i + 2
                while j != S + 1:
                    if "(" in struct[j]:
                        open_count += len(struct[j])
                    elif ")" in struct[j]:
                        close_count += len(struct[j])

                    if open_count == close_count:
                        struct[j] = struct[j] + ")"
                        break
                    j += 1
            return struct

        def set_opening(i, struct):
            close_count = len(struct[i])
            open_count = 0
            j = i - 1
            while j != -1:
                if "(" in struct[j]:
                    open_count += len(struct[j])
                elif ")" in struct[j]:
                    close_count += len(struct[j])
                if open_count == close_count:
                    struct[j] = "(" + struct[j]
                    break
                j -= 1

            return struct

        for comp_score in comp_scores:
            comp_score = np.asarray(comp_score)
            cumulative_comp_score = cumulative_comp_score + comp_score
            #print("step: {}".format(step))
            #print("comp_score: ", comp_score)
            #print("cumulative_comp_score: ", cumulative_comp_score)

            sorted_idx = np.flip(np.argsort(existence_prob * cumulative_comp_score), axis=0).tolist()
            #print(sorted_idx)

            for i in sorted_idx:
                s = struct[i]
                c = cumulative_comp_score[i]
                if existence_prob[i] == 1:
                    if c >= 0.5:
                        if ")" in s:
                            struct = set_opening(i, struct)
                            struct = set_closing(i, struct)
                            existence_prob[i] = 0
                        elif s == "NULL":
                            struct[i] = "("
                            struct = set_closing(i, struct)
                            existence_prob[i] = 0

            #print("existence_prob: ", existence_prob)
            #print("struct: ", struct)
            #print("\n\n")
            step += 1

        new_text = []
        for s, w in zip(struct, text):
            if "(" in s:
                new_text.append(s + w)
            else:
                new_text.append(w + s)

        # print(struct)
        print(" ".join(new_text))


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    config = load_config(args)

    config["encoder"] = "CRvNN_transparent"
    args.model = "CRvNN"

    run(args, config)
