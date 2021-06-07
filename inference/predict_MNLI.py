from parser import get_args
from configs.configLoader import load_config
from utils.path_utils import load_paths
import torch as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.param_utils import param_display_fn, param_count
from utils.checkpoint_utils import load_temp_checkpoint, load_infer_checkpoint, save_infer_checkpoint, \
    save_temp_checkpoint
from utils.data_utils import load_data, load_dataset, load_dataloaders
from controllers.attribute_controller import prepare_attributes
from pathlib import Path
from models import *
from agents import *
from collaters import *
from utils.display_utils import example_display_fn, step_display_fn, display
from trainers import Trainer
from controllers.extras_controller import extras_fn
import csv
from controllers.metric_controller import metric_fn, compose_dev_metric

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

parser = get_args()
args = parser.parse_args()

args.model_type = "NLI"
args.dataset = "MNLI"
args.times = args.initial_time+1


config = load_config(args)

time = args.initial_time

display_string = "Configs:\n"
for k, v in config.items():
    display_string += "{}: {}\n".format(k, v)
display_string += "\n"

paths, checkpoint_paths, metadata = load_paths(args, time)
predi_keys = ["matched", "mismatched"]
paths["predi"] = {key: Path("processed_data/{}_predi_{}.jsonl".format(args.dataset, key))
                 for key in predi_keys}

data = load_data(paths, metadata, args)
idx2labels = data["idx2labels"]
data["predi"] = {key: load_dataset(paths["predi"][key], limit=args.limit) for key in paths["test"]}

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

agent = eval("{}_agent".format(args.model_type))

agent = agent(model=model,
              config=config,
              device=device)

collater = eval("{}_collater".format(args.model_type))
dev_collater = collater(PAD=data["PAD_id"], config=config, train=False)

dataloaders = {}

class Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataloaders["predi"] = {key: DataLoader(Dataset(data["predi"][key]),
                                          batch_size=config["dev_batch_size"] * config["bucket_size_factor"],
                                          num_workers=1,
                                          shuffle=False,
                                          collate_fn=dev_collater.collate_fn) for key in data["predi"]}

agent, epochs_taken = load_infer_checkpoint(agent, checkpoint_paths, paths)
config["current_lr"] = agent.optimizer.param_groups[-1]["lr"]

evaluators = {}
for key in dataloaders["predi"]:
    evaluators[key] = Trainer(config=config,
                              agent=agent,
                              args=args,
                              extras=extras_fn(data, args),
                              logpaths=paths,
                              stats_path=paths["stats_path"],
                              desc="Predicting",
                              sample_len=len(data["predi"][key]),
                              no_display=args.no_display,
                              display_fn=step_display_fn,
                              example_display_fn=example_display_fn)

test_items = {}
test_metric = {}
Path('predictions/').mkdir(parents=True, exist_ok=True)

for key in evaluators:
    test_items[key] = evaluators[key].eval(0, dataloaders["predi"][key])
    display_items = [item["display_items"] for item in test_items[key]]
    csv_path = Path("predictions/MNLI_{}.csv".format(key))
    metrics = [item["metrics"] for item in test_items[key]]
    test_metric[key] = metric_fn(metrics, args)

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['pairID', 'gold_label'])
        for batch_disaplay_items in display_items:
            batch_predictions = batch_disaplay_items["predictions"]
            batch_pairIDs = batch_disaplay_items["pairIDs"]
            for pairID, prediction in zip(batch_pairIDs, batch_predictions):
                writer.writerow([pairID, idx2labels[prediction]])


print(test_metric)

