import jsonlines
import numpy as np

filename = "embeddings/glove/glove.42B.300d.txt"


def load_glove(filename,  vocab=None, dim=300):
    word2embd = {}
    with open(filename, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            contents = line.strip().split()
            word = contents[0].strip()
            if vocab is None or word in vocab:
                embedding = np.asarray(contents[1:])
                if embedding.shape[-1] == dim:
                    word2embd[word] = embedding
    return word2embd


def jsonl_save(filepath, data_dict):
    for key in data_dict:
        data_len = len(data_dict[key])
        break
    objs = []
    for i in range(data_len):
        obj = {key: data_dict[key][i] for key in data_dict}
        objs.append(obj)
    with jsonlines.open(filepath, mode='w') as writer:
        writer.write_all(objs)


# load_glove(filename)
