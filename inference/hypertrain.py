from hyperparser import get_args
from configs.configLoader import load_config
from hyperconfigs.hyperconfigLoader import load_hyperconfig
from hyperopt import fmin, hp, Trials, STATUS_OK
import hyperopt
from train import run
from controllers.metric_controller import compose_dev_metric
import pickle
import numpy as np
from pathlib import Path
import json
import sys
import os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

parser = get_args()
args = parser.parse_args()
search_space, config_processor = load_hyperconfig(args)

max_trials = search_space["max_trials"] if 'max_trials' in search_space else args.max_trials
epochs = search_space["epochs"] if "epochs" in search_space else args.epochs
args.limit = search_space["limit"] if "limit" in search_space else args.limit
hyperalgo = search_space["hyperalgo"] if "hyperalgo" in search_space else args.hyperalgo
allow_repeat = search_space["allow_repeat"] if "allow_repeat" in search_space else args.allow_repeat
args.checkpoint = False
if allow_repeat:
    args.reproducible = False
#print(hyperalgo)
print(search_space)

forbidden_keys = ["max_trials", "epochs", "limit", "hyperalgo", "allow_repeat"]

hp_search_space = {}
for key, val in search_space.items():
    if key not in forbidden_keys:
        hp_search_space[key] = hp.choice(key, val)
space_keys = [k for k in search_space]

hyperopt_config_path = Path("hypertune/tuned_configs/{}_{}.txt".format(args.model, args.dataset))
hyperopt_checkpoint_path = Path("hypertune/checkpoints/{}_{}.pkl".format(args.model, args.dataset))
Path('hypertune/checkpoints/').mkdir(parents=True, exist_ok=True)
Path('hypertune/tuned_configs/').mkdir(parents=True, exist_ok=True)

if args.hypercheckpoint:
    with open(hyperopt_checkpoint_path, "rb") as fp:
        data = pickle.load(fp)
        trials = data["trials"]
        tried_configs = data["tried_configs"]
        true_total_trials = data["true_total_trials"]
    print("\n\nCheckpoint Loaded\n\n")
else:
    trials = Trials()
    tried_configs = {}
    true_total_trials = 0


def generate_args_hash(args):
    hash = ""
    for key in args:
        hash += "{}".format(args[key])
    return hash


successive_failures = 0
max_successive_failures = 10
failure_flag = False


def run_wrapper(space):
    global args
    global tried_configs
    global failure_flag
    config = load_config(args)
    config["epochs"] = epochs
    hash = generate_args_hash(space)

    flag = 0

    if not allow_repeat and hash in tried_configs:
        flag = 1

    if flag == 0:
        print("Exploring: {}".format(space))
        for key in space:
            config[key] = space[key]
        config = config_processor(config)

        blockPrint()
        _, best_metric, _ = run(args, config)
        enablePrint()

        dev_score = compose_dev_metric(best_metric, args, config)
        tried_configs[hash] = -dev_score
        print("loss: {}".format(tried_configs[hash]))
        failure_flag = False
        return {'loss': -dev_score, 'status': STATUS_OK}
    else:
        #print("loss: {} (Skipped Trial)".format(tried_configs[hash]))
        failure_flag = True
        return {'loss': tried_configs[hash], 'status': STATUS_OK}

total_trials = np.prod([len(choices) for key, choices in search_space.items() if key not in forbidden_keys])
max_trials = min(max_trials, total_trials)
save_intervals = 1
i = len(trials.trials)
successive_failures = 0

while True:
    best = fmin(run_wrapper,
                space=hp_search_space,
                algo=eval(hyperalgo), #hyperopt.rand.suggest,
                trials=trials,
                max_evals=len(trials.trials) + save_intervals)

    found_config = {}
    for key in best:
        found_config[key] = search_space[key][best[key]]

    if not failure_flag:
        true_total_trials += 1
        print("Best Config so far: ", found_config)
        print("Total Trials: {} out of {}".format(true_total_trials, max_trials))
        print("\n\n")
        successive_failures = 0
        display_string = ""
        for key, value in found_config.items():
            display_string += "{}: {}\n".format(key, value)
        with open(hyperopt_config_path, "w") as fp:
            fp.write(display_string)

        with open(hyperopt_checkpoint_path, "wb") as fp:
            pickle.dump({"trials": trials,
                         "tried_configs": tried_configs,
                         "true_total_trials": true_total_trials}, fp)
    else:
        successive_failures += 1
        if successive_failures % 1000 == 0:
            print("Successive failures: ", successive_failures)

    if true_total_trials >= max_trials:
        break

    if successive_failures > 100000:
        print("\n\nDiscontinuing due to too many successive failures.\n\n")
        break