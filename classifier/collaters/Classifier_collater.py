import torch as T
import numpy as np
import random

class Classifier_collater:
    def __init__(self, PAD, config, train):
        self.PAD = PAD
        self.config = config
        self.train = train

    def pad(self, items, PAD):
        max_len = max([len(item) for item in items])

        padded_items = []
        item_masks = []
        for item in items:
            mask = [1]*len(item)
            while len(item) < max_len:
                if self.config["left_padded"]:
                    item = [PAD] + item
                    mask = [0] + mask
                else:
                    item.append(PAD)
                    mask.append(0)
            padded_items.append(item)
            item_masks.append(mask)

        return padded_items, item_masks

    def slice_batch(self, batch, start_id, end_id):
        batch_ = {}
        for key in batch:
            batch_[key] = batch[key][start_id:end_id]
        return batch_

    def sort_list(self, objs, idx):
        return [objs[i] for i in idx]


    def collate_fn(self, batch):
        sequences_vec = [obj['sequence_vec'] for obj in batch]
        sequences = [obj['sequence'] for obj in batch]
        labels = [obj['label'] for obj in batch]

        if self.train:
            batch_size = self.config["train_batch_size"]
        else:
            batch_size = self.config["dev_batch_size"]

        bucket_size = len(sequences_vec)

        lengths = [len(obj) for obj in sequences_vec]
        sorted_idx = np.argsort(lengths)

        sequences_vec = self.sort_list(sequences_vec, sorted_idx)
        sequences = self.sort_list(sequences, sorted_idx)
        labels = self.sort_list(labels, sorted_idx)

        batches = []

        i = 0
        while i < bucket_size:
            inr = batch_size
            if i + inr > bucket_size:
                inr = bucket_size - i

            if "stress" not in self.config["dataset"]:
                if self.config["encoder"] != "ordered_memory":
                    max_len = max([len(obj) for obj in sequences_vec[i:i + inr]])
                    if max_len > 25 and max_len <= 70:
                        inr = min(batch_size // 2, inr)
                    elif max_len > 70:
                        inr = min(batch_size // 4, inr)

            sequences_vec_, input_masks = self.pad(sequences_vec[i:i+inr], PAD=self.PAD)

            batch = {}
            batch["sequences_vec"] = T.tensor(sequences_vec_).long()
            batch["sequences"] = sequences[i:i+inr]
            batch["labels"] = T.tensor(labels[i:i+inr]).long()
            batch["input_masks"] = T.tensor(input_masks).float()
            batch["batch_size"] = inr
            batches.append(batch)
            i += inr

        new_batches = []
        new_batch = []
        for j, batch in enumerate(batches):
            if batch["batch_size"] == batch_size // 2:
                new_batch.append(batch)
                if len(new_batch) == 2 or j == len(batches) - 1:
                    new_batches.append(new_batch)
                    new_batch = []
            elif batch["batch_size"] == batch_size // 4:
                new_batch.append(batch)
                if len(new_batch) == 4 or j == len(batches) - 1:
                    new_batches.append(new_batch)
                    new_batch = []
            else:
                new_batches.append([batch])

        if new_batch:
            new_batches.append(new_batch)

        random.shuffle(batches)

        batches = []
        for batch_list in new_batches:
            batches = batches + batch_list


        return batches

