import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import jsonl_save
import csv
import copy

SEED = 101
dev_keys = ["normal"]
test_keys = ["normal"]
np.random.seed(SEED)
random.seed(SEED)
max_seq_len = 100

train_path = Path('../data/Classifier_data/listops/train_d20s.tsv')
test_path = {}
test_path["normal"] = Path('../data/Classifier_data/listops/test_d20s.tsv')

Path('processed_data/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('processed_data/ListOps_train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('processed_data/ListOps_dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('processed_data/ListOps_test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("processed_data/ListOps_metadata.pkl"))

labels2idx = {}
vocab2count = {}


def updateVocab(word):
    global vocab2count
    vocab2count[word] = vocab2count.get(word, 0) + 1


def process_data(filename, update_vocab=True):
    global labels2idx

    print("\n\nOpening directory: {}\n\n".format(filename))

    sequences = []
    labels = []
    count = 0
    with open(filename) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            label = row[0].strip()
            sequence = row[1].strip(" ")
            sequence = sequence.replace("( ","").replace(" )","").split(" ")
            if label not in labels2idx:
                labels2idx[label] = len(labels2idx)
            label_id = labels2idx[label]
            sequences.append(sequence)
            labels.append(label_id)

            print("sequence: ", sequence)
            print("label: ", label)

            if update_vocab:
                for word in sequence:
                    updateVocab(word)

            count += 1

            if count % 1000 == 0:
                print("Processing Data # {}...".format(count))

    return sequences, labels


train_sequences_intermediate, train_labels_intermediate = process_data(train_path)

dev_sequences = {"normal": train_sequences_intermediate[0:1000]}
dev_labels = {"normal": train_labels_intermediate[0:1000]}

train_sequences = []
train_labels = []


for sequence, label in zip(train_sequences_intermediate[1000:], train_labels_intermediate[1000:]):
    if len(sequence) <= max_seq_len:
        train_sequences.append(sequence)
        train_labels.append(label)
"""
for sequence, label in zip(train_sequences_intermediate, train_labels_intermediate):
    if len(sequence) <= max_seq_len:
        train_sequences.append(sequence)
        train_labels.append(label)
"""

test_sequences = {}
test_labels = {}

for key in test_keys:
    test_sequences[key], test_labels[key] = process_data(test_path[key], update_vocab=False)

vocab = [char for char in vocab2count]
vocab += ["<UNK>", "<PAD>", "<SEP>"]

print("train len: ", len(train_sequences))
print("dev len: ", len(dev_sequences["normal"]))
print("test len: ", len(test_sequences["normal"]))
print(np.sort([len(sequence) for sequence in test_sequences["normal"]]))


length_distribution = {}
lengths = []
for sequence in train_sequences:
    length = len(sequence)
    if length in length_distribution:
        length_distribution[length] = length_distribution[length] + 1
    else:
        length_distribution[length] = 1
    lengths.append(length)

lengths = list(set(lengths))
lengths = np.sort(lengths)

for length in lengths:
    print("length {}, samples: {}".format(length, length_distribution[length]))

count = 0
for sequence in test_sequences["normal"]:
    if len(sequence) > 300:
        count += 1

print(count)
print(vocab)
print(labels2idx)

vocab2idx = {word: id for id, word in enumerate(vocab)}


def text_vectorize(text):
    return [vocab2idx[word] for word in text]


def vectorize_data(sequences, labels):
    data_dict = {}
    sequences_vec = [text_vectorize(sequence) for sequence in sequences]
    data_dict["sequence"] = sequences
    data_dict["sequence_vec"] = sequences_vec
    data_dict["label"] = labels
    return data_dict


train_data = vectorize_data(train_sequences,  train_labels)
"""
for item in train_data["sequence1"]:
    print(item)
print("\n\n")
"""
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
