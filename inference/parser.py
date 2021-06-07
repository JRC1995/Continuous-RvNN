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
    parser = ArgumentParser(description="...")
    parser.add_argument('--model', type=str, default="CRvNN",
                        choices=["CRvNN",
                                 "CRvNN2",
                                 "CRvNN_LSTM",
                                 "LR_CRvNN",
                                 "CRvNN_balanced",
                                 "LSTM",
                                 "CRvNN_no_entropy",
                                 "CRvNN_no_gelu",
                                 "CRvNN_no_transition",
                                 "CRvNN_no_modulation",
                                 "ordered_memory"])
    parser.add_argument('--no_display', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--display_params', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--test', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--model_type', type=str, default="NLI",
                        choices=["NLI"])
    parser.add_argument('--dataset', type=str, default="MNLI",
                        choices=["PNLI_LG", "PNLI_A", "PNLI_B", "PNLI_C",
                                 "MNLI", "SNLI"])
    parser.add_argument('--times', type=int, default=5)
    parser.add_argument('--initial_time', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--penalty_gamma', type=float, default=-1)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--example_display_step', type=int, default=500)
    parser.add_argument('--checkpoint', type=str2bool,
                        help="this option can be used to load existing checkpoint",
                        default=False)
    parser.add_argument('--reproducible', type=str2bool, default=True)
    return parser
