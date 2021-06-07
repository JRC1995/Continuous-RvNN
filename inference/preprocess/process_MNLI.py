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
dev_keys = ["matched"]
test_keys = ["matched", "mismatched"]
predi_keys = ["matched", "mismatched"]
np.random.seed(SEED)
random.seed(SEED)

train_path1 = Path('../data/NLI_data/MNLI/multinli_1.0_train.jsonl')
train_path2 = Path('../data/NLI_data/SNLI/snli_1.0_train.jsonl')
dev_path = {}
dev_path["matched"] = Path('../data/NLI_data/MNLI/multinli_1.0_dev_matched.jsonl')
dev_path["mismatched"] = Path('../data/NLI_data/MNLI/multinli_1.0_dev_mismatched.jsonl')
test_path = {}
test_path["matched"] = Path('../data/NLI_data/MNLI/multinli_1.0_dev_matched.jsonl')
test_path["mismatched"] = Path('../data/NLI_data/MNLI/multinli_1.0_dev_mismatched.jsonl')
predi_path = {}
predi_path["matched"] = Path('../data/NLI_data/MNLI/multinli_0.9_test_matched_unlabeled.jsonl')
predi_path["mismatched"] = Path('../data/NLI_data/MNLI/multinli_0.9_test_mismatched_unlabeled.jsonl')
predi2_path = {}
predi2_path["matched"] = Path(
    '../data/NLI_data/MNLI/multinli_1.0_dev_matched.jsonl')  # Path('../../data/NLI_data/MNLI/multinli_0.9_test_matched_unlabeled.jsonl')
predi2_path["mismatched"] = Path(
    '../data/NLI_data/MNLI/multinli_1.0_dev_mismatched.jsonl')  # Path('../../data/NLI_data/MNLI/multinli_0.9_test_mismatched_unlabeled.jsonl')

embedding_path = Path("../embeddings/glove/glove.840B.300d.txt")

Path('processed_data/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('processed_data/MNLI_train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('processed_data/MNLI_dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('processed_data/MNLI_test_{}.jsonl'.format(key))
predi_save_path = {}
predi2_save_path = {}
for key in predi_keys:
    predi_save_path[key] = Path('processed_data/MNLI_predi_{}.jsonl'.format(key))
    predi2_save_path[key] = Path('processed_data/MNLI_predi2_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("processed_data/MNLI_metadata.pkl"))

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
    pairIDs = []
    labels = []
    count = 0
    max_seq_len = 150

    with jsonlines.open(filename) as reader:
        for sample in reader:
            if sample['gold_label'] != '-':

                sequence1 = tokenize(sample['sentence1'].lower())
                sequence2 = tokenize(sample['sentence2'].lower())
                pairID = sample["pairID"]
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
                        pairIDs.append(pairID)
                else:
                    sequences1.append(sequence1)
                    sequences2.append(sequence2)
                    labels.append(label_id)
                    pairIDs.append(pairID)

                if update_vocab:
                    for word in sequence1:
                        updateVocab(word)

                    for word in sequence2:
                        updateVocab(word)

                count += 1

                if count % 1000 == 0:
                    print("Processing Data # {}...".format(count))

    return sequences1, sequences2, labels, pairIDs


train_sequences1, \
train_sequences2, \
train_labels, _ = process_data(train_path1, filter=True)

train_sequences1_, \
train_sequences2_, \
train_labels_, _ = process_data(train_path2, filter=True)

train_sequences1 += train_sequences1_
train_sequences2 += train_sequences2_
train_labels += train_labels_

dev_sequences1 = {}
dev_sequences2 = {}
dev_labels = {}

for key in dev_keys:
    dev_sequences1[key], \
    dev_sequences2[key], \
    dev_labels[key], _ = process_data(dev_path[key], update_vocab=True)

test_sequences1 = {}
test_sequences2 = {}
test_labels = {}

for key in test_keys:
    test_sequences1[key], \
    test_sequences2[key], \
    test_labels[key], _ = process_data(test_path[key], update_vocab=True)

predi_sequences1 = {}
predi_sequences2 = {}
predi_labels = {}
predi_pairIDs = {}

for key in predi_keys:
    predi_sequences1[key], \
    predi_sequences2[key], \
    predi_labels[key], predi_pairIDs[key] = process_data(predi_path[key], update_vocab=True)

predi2_sequences1 = {}
predi2_sequences2 = {}
predi2_labels = {}
predi2_pairIDs = {}

for key in predi_keys:
    predi2_sequences1[key], \
    predi2_sequences2[key], \
    predi2_labels[key], predi2_pairIDs[key] = process_data(predi2_path[key], update_vocab=False)

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


def vectorize_data(sequences1, sequences2, labels, pairIDs=None):
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
    if pairIDs is not None:
        data_dict["pairID"] = pairIDs
        print(data_dict["pairID"])
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

predi_data = {}
for key in predi_keys:
    predi_data[key] = vectorize_data(predi_sequences1[key], predi_sequences2[key], predi_labels[key],
                                     predi_pairIDs[key])

predi2_data = {}
for key in predi_keys:
    predi2_data[key] = vectorize_data(predi2_sequences1[key], predi2_sequences2[key], predi2_labels[key],
                                      predi2_pairIDs[key])

jsonl_save(filepath=train_save_path,
           data_dict=train_data)

for key in dev_keys:
    jsonl_save(filepath=dev_save_path[key],
               data_dict=dev_data[key])

for key in test_keys:
    jsonl_save(filepath=test_save_path[key],
               data_dict=test_data[key])

for key in predi_keys:
    jsonl_save(filepath=predi_save_path[key],
               data_dict=predi_data[key])
    jsonl_save(filepath=predi2_save_path[key],
               data_dict=predi2_data[key])


metadata = {"labels2idx": labels2idx,
            "vocab2idx": vocab2idx,
            "embeddings": np.asarray(embeddings, np.float32),
            "dev_keys": dev_keys,
            "test_keys": test_keys}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)
