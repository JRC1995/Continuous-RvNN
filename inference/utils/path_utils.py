import pickle
from pathlib import Path


def load_paths(args, time=0):
    if args.model_type == "NLI" or args.model_type == "Classifier":
        metadata_path = Path("processed_data/{}_metadata.pkl".format(args.dataset))
        with open(metadata_path, 'rb') as fp:
            metadata = pickle.load(fp)
        dev_keys = metadata["dev_keys"]
        test_keys = metadata["test_keys"]
        paths = {}
        paths["train"] = Path("processed_data/{}_train.jsonl".format(args.dataset))
        paths["dev"] = {key: Path("processed_data/{}_dev_{}.jsonl".format(args.dataset, key))
                        for key in dev_keys}
        paths["test"] = {key: Path("processed_data/{}_test_{}.jsonl".format(args.dataset, key))
                         for key in test_keys}

        test_flag = "_test" if args.test else ""

        paths["verbose_log_path"] = Path(
            "experiments/logs/{}_{}_{}/{}_verbose_logs{}.txt".format(args.dataset, args.model, args.model_type, time,
                                                                     test_flag))
        paths["log_path"] = Path(
            "experiments/logs/{}_{}_{}/{}_logs{}.txt".format(args.dataset, args.model, args.model_type, time,
                                                             test_flag))
        paths["stats_path"] = Path(
            "experiments/logs/{}_{}_{}/{}_stats{}.txt".format(args.dataset, args.model, args.model_type, time,
                                                              test_flag))



        Path('experiments/checkpoints').mkdir(parents=True, exist_ok=True)
        Path('experiments/logs/{}_{}_{}'.format(args.dataset, args.model, args.model_type)).mkdir(parents=True,
                                                                                                  exist_ok=True)
        Path('saved_weights/{}_{}_{}'.format(args.dataset, args.model, args.model_type)).mkdir(parents=True,
                                                                                               exist_ok=True)

        if not args.checkpoint:
            with open(paths["verbose_log_path"], "w+") as fp:
                pass
            with open(paths["log_path"], "w+") as fp:
                pass
            with open(paths["stats_path"], "w+") as fp:
                pass

        checkpoint_paths = {"infer_checkpoint_path": Path(
            "saved_weights/{}_{}_{}/{}.pt".format(args.dataset, args.model, args.model_type, time)),
                            "temp_checkpoint_path": Path(
                                "experiments/checkpoints/{}_{}_{}.pt".format(args.dataset, args.model,
                                                                             args.model_type))}

        return paths, checkpoint_paths, metadata
