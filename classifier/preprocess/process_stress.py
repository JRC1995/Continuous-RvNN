import random
import numpy as np
from pathlib import Path
from os import fspath
import copy
import pickle
from preprocess_tools.process_utils import jsonl_save

MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SM"
END = "]"

random.seed(101)

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

VALUE_P = 0.25
MAX_ARGS = 5
MAX_DEPTH = 20

DATA_POINTS = 100000


def generate_tree(depth):
    if depth < MAX_DEPTH:
        r = random.random()
    else:
        r = 1

    if r > VALUE_P:
        value = random.choice(VALUES)
        return value
    else:
        num_values = random.randint(2, MAX_ARGS)
        values = []
        for _ in range(num_values):
            values.append(generate_tree(depth + 1))

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t


def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'


def to_value(t):
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == FIRST:
            return l[1][0]
        elif l[0] == LAST:
            return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return (np.sum(l[1]) % 10)
    elif isinstance(l, tuple):  # We've hit an unsaturated function and an argument.
        return (l[0], l[1] + [r])


max_samples = 50
dataset = {}
all_examples = {}
keys = ["100", "200", "500", "700", "1000"]
while True:
    example = generate_tree(1)
    if example not in all_examples:
        all_examples[example] = 1
        sample = to_string(example).replace("( ", "").replace(" )", "").split(" ")
        label = int(to_value(example))
        if len(sample) > 80 and len(sample) <= 100:
            key = "100"
        elif len(sample) > 100 and len(sample) <= 200:
            key = "200"
        elif len(sample) > 200 and len(sample) <= 500:
            key = "500"
        elif len(sample) > 500 and len(sample) <= 700:
            key = "700"
        elif len(sample) > 700 and len(sample) <= 1000:
            key = "1000"
        else:
            continue

        if len(dataset.get(key, [])) < max_samples:
            dataset[key] = dataset.get(key, []) + [{"sample": sample, "label": label}]

        flag = True
        for key in keys:
            if len(dataset.get(key, [])) < max_samples:
                flag = False

        if flag:
            break

labels2idx = {str(id): id for id in range(10)}
vocab = [str(id) for id in range(10)] + [MIN, MAX, MED, FIRST, LAST, SUM_MOD, END, "<PAD>"]
vocab2idx = {token: i for i, token in enumerate(vocab)}


def text_vectorize(text):
    return [vocab2idx[word] for word in text]


def vectorize_data(sequences, labels):
    data_dict = {}
    sequences_vec = [text_vectorize(sequence) for sequence in sequences]
    data_dict["sequence"] = sequences
    data_dict["sequence_vec"] = sequences_vec
    data_dict["label"] = labels
    print(labels)
    return data_dict

Path('processed_data/').mkdir(parents=True, exist_ok=True)
dev_keys = ["normal"]
test_keys = ["normal"]
for key_ in keys:
    train_save_path = Path('processed_data/stress{}_train.jsonl'.format(key_))
    dev_save_path = {}
    for key in dev_keys:
        dev_save_path[key] = Path('processed_data/stress{}_dev_{}.jsonl'.format(key_,key))
    test_save_path = {}
    for key in test_keys:
        test_save_path[key] = Path('processed_data/stress{}_test_{}.jsonl'.format(key_,key))
    metadata_save_path = fspath(Path("processed_data/stress{}_metadata.pkl".format(key_)))

    train_sequences = [item["sample"] for item in dataset[key_]]
    train_labels = [item["label"] for item in dataset[key_]]

    dev_sequences = {}
    dev_labels = {}
    dev_sequences["normal"] = copy.deepcopy(train_sequences)
    dev_labels["normal"] = copy.deepcopy(train_labels)

    test_sequences = {}
    test_labels = {}
    test_sequences["normal"] = copy.deepcopy(train_sequences)
    test_labels["normal"] = copy.deepcopy(train_labels)

    train_data = vectorize_data(train_sequences, train_labels)
    dev_data = {}
    for key in dev_keys:
        dev_data[key] = vectorize_data(dev_sequences[key], dev_labels[key])
    test_data = {}
    for key in test_keys:
        test_data[key] = vectorize_data(test_sequences[key], test_labels[key])

    jsonl_save(filepath=train_save_path,
               data_dict=train_data)

    for key in dev_keys:
        jsonl_save(filepath=dev_save_path[key],
                   data_dict=dev_data[key])

    for key in test_keys:
        jsonl_save(filepath=test_save_path[key],
                   data_dict=test_data[key])

    metadata = {"labels2idx": labels2idx,
                "vocab2idx": vocab2idx,
                "dev_keys": dev_keys,
                "test_keys": test_keys}

    with open(metadata_save_path, 'wb') as outfile:
        pickle.dump(metadata, outfile)
