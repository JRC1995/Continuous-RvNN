import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
import re
from preprocess_tools.process_utils import jsonl_save

SEED = 101
train_keys = [str(i) for i in range(7)]
dev_keys = ["normal"]
test_keys = ["normal"]
regex_expression = '.*\( (and|or) \( not .* \) \).*'
regex_expression = re.compile(regex_expression)
np.random.seed(SEED)
random.seed(SEED)

train_path = {}
for key in train_keys:
    train_path[key] = Path('../data/NLI_data/PNLI/train{}'.format(key))


Path('processed_data/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('processed_data/PNLI_C_train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('processed_data/PNLI_C_dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('processed_data/PNLI_C_test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("processed_data/PNLI_C_metadata.pkl"))

labels2idx = {}
vocab2count = {}


def updateVocab(word):
    global vocab2count
    vocab2count[word] = vocab2count.get(word, 0)+1


def process_data(filename, update_vocab=True):

    global labels2idx

    print("\n\nOpening directory: {}\n\n".format(filename))

    sequences1 = []
    sequences2 = []
    labels = []
    count = 0

    with open(filename) as reader:
        lines = reader.readlines()
        for sample in lines:
            sample = sample.strip().split("\t")
            label = sample[0]
            sequence1 = sample[1].split(" ")
            sequence2 = sample[2].split(" ")
            if label not in labels2idx:
                labels2idx[label] = len(labels2idx)
            label_id = labels2idx[label]

            sequences1.append(sequence1)
            sequences2.append(sequence2)
            labels.append(label_id)

            #print("sequence1: ", sequence1)
            #print("sequence2: ", sequence2)
            #print("label: ", label)


            if update_vocab:
                for word in sequence1:
                    updateVocab(word)

                for word in sequence2:
                    updateVocab(word)

            count += 1

            if count % 1000 == 0:
                print("Processing Data # {}...".format(count))

    return sequences1, sequences2, labels

train_sequences1 = []
train_sequences2 = []
train_labels = []
dev_sequences1 = {k: [] for k in dev_keys}
dev_sequences2 = {k: [] for k in dev_keys}
dev_labels = {k: [] for k in dev_keys}
test_sequences1 = {k: [] for k in test_keys}
test_sequences2 = {k: [] for k in test_keys}
test_labels = {k: [] for k in test_keys}

counter = 0
for key in train_keys:
    sequences1, sequences2, labels = process_data(train_path[key])
    for sequence1, sequence2, label in zip(sequences1, sequences2, labels):
        if regex_expression.match(" ".join(sequence1)) or regex_expression.match(" ".join(sequence2)):
            test_sequences1["normal"].append(sequence1)
            test_sequences2["normal"].append(sequence2)
            test_labels["normal"].append(label)
        else:
            counter += 1
            if counter % 10 == 0:
                dev_sequences1["normal"].append(sequence1)
                dev_sequences2["normal"].append(sequence2)
                dev_labels["normal"].append(label)
            else:
                train_sequences1.append(sequence1)
                train_sequences2.append(sequence2)
                train_labels.append(label)


vocab = [char for char in vocab2count]
vocab += ["<UNK>", "<PAD>", "<SEP>"]

print("train len: ", len(train_sequences1))
print("dev len: ", len(dev_sequences1["normal"]))
print("test len: ", len(test_sequences1["normal"]))
print(np.sort([len(sequence) for sequence in train_sequences1]))

print(vocab)
print(labels2idx)

vocab2idx = {word: id for id, word in enumerate(vocab)}

def text_vectorize(text):
    return [vocab2idx[word] for word in text]


def vectorize_data(sequences1, sequences2, labels):
    data_dict = {}
    sequences1_vec = [text_vectorize(sequence) for sequence in sequences1]
    sequences2_vec = [text_vectorize(sequence) for sequence in sequences2]
    sequences_vec = [sequence1 + [vocab2idx["<SEP>"]] + sequence2 for sequence1, sequence2 in zip(sequences1_vec, sequences2_vec)]
    data_dict["sequence1"] = sequences1
    data_dict["sequence2"] = sequences2
    data_dict["sequence1_vec"] = sequences1_vec
    data_dict["sequence2_vec"] = sequences2_vec
    data_dict["sequence_vec"] = sequences_vec
    data_dict["label"] = labels
    return data_dict


train_data = vectorize_data(train_sequences1, train_sequences2, train_labels)
"""
for item in train_data["sequence1"]:
    print(item)
print("\n\n")
"""
dev_data = {}
for key in dev_keys:
    dev_data[key] = vectorize_data(dev_sequences1[key], dev_sequences2[key], dev_labels[key])
test_data = {}
for key in test_keys:
    test_data[key] = vectorize_data(test_sequences1[key], test_sequences2[key], test_labels[key])

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
