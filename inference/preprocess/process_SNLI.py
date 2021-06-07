import math
import pickle
import random
from os import fspath
from pathlib import Path

import jsonlines
import nltk
import numpy as np

from preprocess_tools.process_utils import load_glove, jsonl_save

SEED = 101
MAX_VOCAB = 50000
MIN_FREQ = 1
WORDVECDIM = 300
dev_keys = ["normal"]
test_keys = ["normal", "hard"]
np.random.seed(SEED)
random.seed(SEED)

train_path = Path('../data/NLI_data/SNLI/snli_1.0_train.jsonl')
dev_path = {}
dev_path["normal"] = Path('../data/NLI_data/SNLI/snli_1.0_dev.jsonl')
test_path = {}
test_path["normal"] = Path('../data/NLI_data/SNLI/snli_1.0_test.jsonl')
test_path["hard"] = Path('../data/NLI_data/SNLI/snli_1.0_test_hard.jsonl')


embedding_path = Path("../embeddings/glove/glove.840B.300d.txt")

Path('processed_data/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('processed_data/SNLI_train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('processed_data/SNLI_dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('processed_data/SNLI_test_{}.jsonl'.format(key))

metadata_save_path = fspath(Path("processed_data/SNLI_metadata.pkl"))

labels2idx = {}
vocab2count = {}


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def updateVocab(word):
    global vocab2count
    vocab2count[word] = vocab2count.get(word, 0) + 1


def process_data(filename, update_vocab=True, filter=False, predi=False):
    global labels2idx

    print("\n\nOpening directory: {}\n\n".format(filename))

    sequences1 = []
    sequences2 = []
    labels = []
    count = 0
    max_seq_len = 150

    with jsonlines.open(filename) as reader:
        for sample in reader:
            if sample['gold_label'] != '-':

                sequence1 = tokenize(sample['sentence1'].lower())
                sequence2 = tokenize(sample['sentence2'].lower())
                if predi:
                    label = None
                    label_id = None
                else:
                    label = sample['gold_label']
                    if label not in labels2idx:
                        labels2idx[label] = len(labels2idx)
                    label_id = labels2idx[label]

                if filter:
                    if (len(sequence1) < max_seq_len) and (len(sequence2) < max_seq_len):
                        sequences1.append(sequence1)
                        sequences2.append(sequence2)
                        labels.append(label_id)
                else:
                    sequences1.append(sequence1)
                    sequences2.append(sequence2)
                    labels.append(label_id)

                if update_vocab:
                    for word in sequence1:
                        updateVocab(word)

                    for word in sequence2:
                        updateVocab(word)

                count += 1

                if count % 1000 == 0:
                    print("Processing Data # {}...".format(count))

    return sequences1, sequences2, labels


train_sequences1, \
train_sequences2, \
train_labels = process_data(train_path, filter=True)

dev_sequences1 = {}
dev_sequences2 = {}
dev_labels = {}

for key in dev_keys:
    dev_sequences1[key], \
    dev_sequences2[key], \
    dev_labels[key] = process_data(dev_path[key], update_vocab=True)

test_sequences1 = {}
test_sequences2 = {}
test_labels = {}

for key in test_keys:
    test_sequences1[key], \
    test_sequences2[key], \
    test_labels[key] = process_data(test_path[key], update_vocab=True)

counts = []
vocab = []
for word, count in vocab2count.items():
    if count > MIN_FREQ:
        vocab.append(word)
        counts.append(count)

vocab2embed = load_glove(embedding_path, vocab=vocab2count, dim=WORDVECDIM)

sorted_idx = np.flip(np.argsort(counts), axis=0)
vocab = [vocab[id] for id in sorted_idx if vocab[id] in vocab2embed]
if len(vocab) > MAX_VOCAB:
    vocab = vocab[0:MAX_VOCAB]

vocab += ["<PAD>", "<UNK>", "<SEP>"]

print(vocab)

vocab2idx = {word: id for id, word in enumerate(vocab)}

vocab2embed["<PAD>"] = np.zeros((WORDVECDIM), np.float32)
b = math.sqrt(3 / WORDVECDIM)
vocab2embed["<UNK>"] = np.random.uniform(-b, +b, WORDVECDIM)
vocab2embed["<SEP>"] = np.random.uniform(-b, +b, WORDVECDIM)

embeddings = []
for id, word in enumerate(vocab):
    embeddings.append(vocab2embed[word])


def text_vectorize(text):
    return [vocab2idx.get(word, vocab2idx['<UNK>']) for word in text]


def vectorize_data(sequences1, sequences2, labels):
    data_dict = {}
    sequences1_vec = [text_vectorize(sequence) for sequence in sequences1]
    sequences2_vec = [text_vectorize(sequence) for sequence in sequences2]
    data_dict["sequence1"] = sequences1
    data_dict["sequence2"] = sequences2
    sequences_vec = [sequence1 + [vocab2idx["<SEP>"]] + sequence2 for sequence1, sequence2 in
                     zip(sequences1_vec, sequences2_vec)]
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
            "embeddings": np.asarray(embeddings, np.float32),
            "dev_keys": dev_keys,
            "test_keys": test_keys}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)
