import math
import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
import torchtext
from torchtext import datasets
from preprocess_tools.process_utils import load_glove, jsonl_save

SEED = 101
MAX_VOCAB = 50000
MIN_FREQ = 1
WORDVECDIM = 300
dev_keys = ["normal"]
test_keys = ["normal"]
np.random.seed(SEED)
random.seed(SEED)

embedding_path = Path("../embeddings/glove/glove.840B.300d.txt")
Path('processed_data/').mkdir(parents=True, exist_ok=True)


train_save_path = Path('processed_data/SST5_train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('processed_data/SST5_dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('processed_data/SST5_test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("processed_data/SST5_metadata.pkl"))

labels2idx = {}
vocab2count = {}

def tokenize(sequence):
    return sequence.split()


def updateVocab(word):
    global vocab2count
    vocab2count[word] = vocab2count.get(word, 0)+1


def process_data(dataset, update_vocab=True):

    global labels2idx

    sequences = []
    labels = []
    count = 0

    for item in dataset:

        sequence = item.text
        sequence = [token.lower() for token in sequence]
        label = item.label

        if label not in labels2idx:
            labels2idx[label] = len(labels2idx)
        label_id = labels2idx[label]

        sequences.append(sequence)
        labels.append(label_id)

        if update_vocab:
            for word in sequence:
                updateVocab(word)

        count += 1

        if count % 1000 == 0:
            print("Processing Data # {}...".format(count))

    return sequences, labels

TEXT = torchtext.data.Field(lower=True, include_lengths=False, batch_first=True)
LABEL = torchtext.data.Field(sequential=False, unk_token=None)

# make splits for data
filter_pred = None
fine_grained = True

"""
if args.dataset == "SST5":
    fine_grained = True
else:
    fine_grained = False

if not args.fine_grained:
    filter_pred = lambda ex: ex.label != 'neutral'
"""
train_set, dev_set, test_set = datasets.SST.splits(
    TEXT, LABEL,
    train_subtrees=True,
    fine_grained=True,
    filter_pred=None
)


train_sequences, train_labels = process_data(train_set)

print("training size: ", len(train_sequences))


dev_sequences = {}
dev_labels = {}

for key in dev_keys:
    dev_sequences[key], dev_labels[key] = process_data(dev_set)
    print("Development size (key:{})".format(key), len(dev_sequences[key]))


test_sequences = {}
test_labels = {}

for key in test_keys:
    test_sequences[key], test_labels[key] = process_data(test_set)
    print("Test size (key:{})".format(key), len(test_sequences[key]))

print(len(vocab2count))
counts = []
vocab = []
for word, count in vocab2count.items():
    if count >= MIN_FREQ:
        vocab.append(word)
        counts.append(count)
print(len(vocab))


vocab2embed = load_glove(embedding_path, vocab=vocab2count, dim=WORDVECDIM)

print(len(vocab2embed))

sorted_idx = np.flip(np.argsort(counts), axis=0)
vocab = [vocab[id] for id in sorted_idx if vocab[id] in vocab2embed]
if len(vocab) > MAX_VOCAB:
    vocab = vocab[0:MAX_VOCAB]

vocab += ["<PAD>", "<UNK>", "<SEP>"]

#print(vocab)

vocab2idx = {word: id for id, word in enumerate(vocab)}

vocab2embed["<PAD>"] = np.zeros((WORDVECDIM), np.float32)
b = math.sqrt(3/WORDVECDIM)
vocab2embed["<UNK>"] = np.random.uniform(-b, +b, WORDVECDIM)
vocab2embed["<SEP>"] = np.random.uniform(-b, +b, WORDVECDIM)

embeddings = []
for id, word in enumerate(vocab):
    embeddings.append(vocab2embed[word])


def text_vectorize(text):
    return [vocab2idx.get(word, vocab2idx['<UNK>']) for word in text]


def vectorize_data(sequences, labels):
    data_dict = {}
    sequences_vec = [text_vectorize(sequence) for sequence in sequences]
    data_dict["sequence"] = sequences
    data_dict["sequence_vec"] = sequences_vec
    data_dict["label"] = labels
    return data_dict

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
            "embeddings": np.asarray(embeddings, np.float32),
            "dev_keys": dev_keys,
            "test_keys": test_keys}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)
