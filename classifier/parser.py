import argparse
from argparse import ArgumentParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser(description="....")
    parser.add_argument('--model', type=str, default="FOCN",
                        choices=["FOCN",
                                 "FOCN_LSTM",
                                 "LR_FOCN",
                                 "FOCN_no_entropy",
                                 "FOCN_no_gelu",
                                 "FOCN_no_recurrency_bias",
                                 "FOCN_no_modulation",
                                 "ordered_memory"])
    parser.add_argument('--no_display', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--display_params', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--test', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--model_type', type=str, default="Classifier",
                        choices=["Classifier"])
    parser.add_argument('--dataset', type=str, default="ListOps",
                        choices=["ListOps",
                                 "stress100", "stress200", "stress500","stress700", "stress1000",
                                 "SST5",  "SST2"])
    parser.add_argument('--times', type=int, default=5)
    parser.add_argument('--initial_time', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--penalty_gamma', type=float, default=-1)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--example_display_step', type=int, default=500)
    parser.add_argument('--checkpoint', type=str2bool,
                        help="this option can be used to load existing checkpoint",
                        default=False)
    return parser
