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
        #T.set_deterministic(True)

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

    collater = eval("{}_collater".format(args.model_type))
    train_collater = collater(PAD=data["PAD_id"], config=config, train=True)
    dev_collater = collater(PAD=data["PAD_id"], config=config, train=False)

    dataloaders = load_dataloaders(train_batch_size=config["train_batch_size"] * config["bucket_size_factor"],
                                   dev_batch_size=config["dev_batch_size"] * config["bucket_size_factor"],
                                   partitions=data,
                                   train_collater_fn=train_collater.collate_fn,
                                   dev_collater_fn=dev_collater.collate_fn)

    epochs_taken = 0

    if not args.test:

        agent, loaded_stuff = load_temp_checkpoint(agent, time, checkpoint_paths, args, paths)
        config["current_lr"] = agent.optimizer.param_groups[-1]["lr"]
        time = loaded_stuff["time"]
        _, checkpoint_paths, _ = load_paths(args, time)
        if loaded_stuff["random_states"] is not None:
            random_states = loaded_stuff["random_states"]
            random.setstate(random_states["python_random_state"])
            np.random.set_state(random_states["np_random_state"])
            T.random.set_rng_state(random_states["torch_random_state"])

        epochs = config["epochs"]
        trainer = Trainer(config=config,
                          agent=agent,
                          args=args,
                          extras=extras_fn(data, args),
                          logpaths=paths,
                          desc="Training",
                          sample_len=len(data["train"]),
                          global_step=loaded_stuff["global_step"],
                          no_display=args.no_display,
                          display_fn=step_display_fn,
                          example_display_fn=example_display_fn)

        evaluators = {}
        for key in dataloaders["dev"]:
            evaluators[key] = Trainer(config=config,
                                      agent=agent,
                                      args=args,
                                      extras=extras_fn(data, args),
                                      logpaths=paths,
                                      desc="Validating",
                                      sample_len=len(data["dev"][key]),
                                      no_display=args.no_display,
                                      display_fn=step_display_fn,
                                      example_display_fn=example_display_fn)

        past_epochs = loaded_stuff["past_epochs"]

        for epoch in range(past_epochs, epochs):

            if loaded_stuff["impatience"] > config["early_stop_patience"]:
                break

            display("\nRun {}; Training Epoch # {}\n".format(time, epoch), paths)

            train_items = trainer.train(epoch, dataloaders["train"])
            metrics = [item["metrics"] for item in train_items]
            train_metric = metric_fn(metrics, args)

            display("\nRun {}; Validating Epoch # {}\n".format(time, epoch), paths)

            dev_items = {}
            dev_metric = {}

            for key in evaluators:
                dev_items[key] = evaluators[key].eval(epoch, dataloaders["dev"][key])
                metrics = [item["metrics"] for item in dev_items[key]]
                dev_metric[key] = metric_fn(metrics, args)

            dev_score = compose_dev_metric(dev_metric, args, config)

            loaded_stuff["past_epochs"] += 1

            if agent.epoch_level_scheduler:
                agent.scheduler.step(dev_score)

            display_string = "\n\nEpoch {} Summary:\n".format(epoch)
            display_string += "Training "
            for k, v in train_metric.items():
                display_string += "{}: {}; ".format(k, v)
            display_string += "\n\n"

            for key in dev_metric:
                display_string += "Validation ({}) ".format(key)
                for k, v in dev_metric[key].items():
                    display_string += "{}: {}; ".format(k, v)
                display_string += "\n"

            display_string += "\n"

            display(display_string, paths)

            loaded_stuff["impatience"] += 1


            if (dev_score - loaded_stuff["best_dev_score"]) > 0.0001:
                loaded_stuff["best_dev_score"] = dev_score
                loaded_stuff["best_dev_metric"] = dev_metric
                loaded_stuff["impatience"] = 0
                epochs_taken = epoch + 1
                save_infer_checkpoint(epoch+1, agent, checkpoint_paths, paths)

            display_string = "\nImpatience: {}\n".format(loaded_stuff["impatience"])
            display(display_string, paths)

            loaded_stuff["random_states"] = {'python_random_state': random.getstate(),
                                             'np_random_state': np.random.get_state(),
                                             'torch_random_state': T.random.get_rng_state()}

            save_temp_checkpoint(agent, checkpoint_paths, loaded_stuff, paths)

            if loaded_stuff["impatience"] > config["early_stop_patience"]:
                break

        return time, loaded_stuff["best_dev_metric"], epochs_taken

    else:
        agent, epochs_taken = load_infer_checkpoint(agent, checkpoint_paths, paths)
        config["current_lr"] = agent.optimizer.param_groups[-1]["lr"]

        evaluators = {}
        for key in dataloaders["test"]:
            evaluators[key] = Trainer(config=config,
                                      agent=agent,
                                      args=args,
                                      extras=extras_fn(data, args),
                                      logpaths=paths,
                                      stats_path=paths["stats_path"],
                                      desc="Testing",
                                      sample_len=len(data["test"][key]),
                                      no_display=args.no_display,
                                      display_fn=step_display_fn,
                                      example_display_fn=example_display_fn)

        display("\nTesting\n", paths)
        display("\nEpochs Taken: {}\n".format(epochs_taken), paths)

        test_items = {}
        test_metric = {}

        for key in evaluators:
            test_items[key] = evaluators[key].eval(0, dataloaders["test"][key])
            metrics = [item["metrics"] for item in test_items[key]]
            test_metric[key] = metric_fn(metrics, args)

        display_string = ""

        for key in test_metric:
            display_string += "Test ({}) ".format(key)
            for k, v in test_metric[key].items():
                display_string += "{}: {}; ".format(k, v)
            display_string += "\n"

        display_string += "\n"

        display(display_string, paths)

        return time, test_metric, epochs_taken

def run_and_collect_results(args, config):

    best_metrics = {}

    test_flag = "_test" if args.test else ""
    final_result_path = Path(
        "experiments/final_results/{}_{}_{}{}.txt".format(args.dataset, args.model, args.model_type, test_flag))
    Path('experiments/final_results').mkdir(parents=True, exist_ok=True)

    time = args.initial_time
    while time < args.times:
        if time != args.initial_time:
            args.checkpoint = False
        time, best_metric, epochs_taken = run(args, config, time)

        for key in best_metric:
            if key in best_metrics:
                for k, v in best_metric[key].items():
                    if k in best_metrics[key]:
                        best_metrics[key][k].append(v)
                    else:
                        best_metrics[key][k] = [v]
            else:
                best_metrics[key] = {}
                for k, v in best_metric[key].items():
                    best_metrics[key][k] = [v]

            if "epochs_taken" in best_metrics[key]:
                best_metrics[key]["epochs_taken"].append(epochs_taken)
            else:
                best_metrics[key]["epochs_taken"] = [epochs_taken]

        display_string = "\n\nBest of Run {} (Epochs Taken: {}):\n".format(time, epochs_taken)

        for key in best_metric:
            display_string += "({}) ".format(key)
            for k, v in best_metric[key].items():
                display_string += "{}: {}; ".format(k, v)
            display_string += "\n"
        display_string += "\n"

        print(display_string)

        if time == 0:
            mode = "w"
        else:
            mode = "a"
        with open(final_result_path, mode) as fp:
            fp.write(display_string)

        time += 1

    display_string = "\n\nMean\Std:\n\n"
    for key in best_metrics:
        display_string += "({}) ".format(key)
        for k, v in best_metrics[key].items():
            display_string += "{}: {} (max) {} (median) {} (mean) +- {} (std); ".format(k, max(v), np.median(v), np.mean(v), np.std(v))
        display_string += "\n"

    print(display_string)
    with open(final_result_path, "a") as fp:
        fp.write(display_string)





if __name__ == '__main__':

    parser = get_args()
    args = parser.parse_args()
    config = load_config(args)

    run_and_collect_results(args, config)

    if not args.test:
        args.test = True
        run_and_collect_results(args, config)
