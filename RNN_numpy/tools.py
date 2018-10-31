import numpy as np


def read_data(path='datasets/data.txt'):
    data = open(path, 'r').read()
    chars = list(set(data))
    return data, chars


def convert_to_one_hot(data_, vocab):
    data = np.zeros((len(data_), len(vocab)))
    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1
    return data


def load_data(input):
    # Load the data
    data_ = ""
    with open(input, 'r') as f:
        data_ += f.read()
    data_ = data_.lower()
    # Convert to 1-hot coding
    vocab = sorted(list(set(data_)))
    data = convert_to_one_hot(data_, vocab)
    return data, vocab
