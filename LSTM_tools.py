import numpy as np


def decode_embed(array, vocab):
    return vocab[array.index(1)]


# main characters: 0-127 = 0-127
# persian characters: (char code) 1560-1751 <-> (id) 128-319
# <start> = 320, <end> = 321, <unk> = 322
START = 320
END = 321
UNK = 322
SIZE_OF_VOCAB = 323


def map_char_to_id(c):
    code = ord(c)
    if code < 128:
        return code
    if 1560 <= code <= 1751:
        return code - 1432
    return UNK


def map_id_to_char(code):
    if code == START: return '<START>'
    if code == END: return '<END>'
    if code == UNK: return '<UNK>'
    if code < 128: return chr(code)
    if 128 <= code <= 319: return chr(code + 1432)
    return '<?>'


def convert_to_one_hot_old(data_, vocab): # todo: delete ir
    data = np.zeros((len(data_), len(vocab)))
    for cnt, s in enumerate(data_):
        v = [0.0] * len(vocab)
        print(s)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
    return data


def convert_to_one_hot(sentence, char_num_of_sentence):
    s_vector = np.zeros((char_num_of_sentence, SIZE_OF_VOCAB))
    for char_id, charac in enumerate(sentence):
        v = [0.0] * SIZE_OF_VOCAB
        v[map_char_to_id(charac)] = 1.0
        s_vector[char_id+1] = v

    v = [0.0] * SIZE_OF_VOCAB
    v[START] = 1.0
    s_vector[0] = v
    v = [0.0] * SIZE_OF_VOCAB
    v[END] = 1.0
    s_vector[char_num_of_sentence-1] = v

    return s_vector


def load_data(input, char_num_of_sentence):
    # Load the data
    with open(input, 'r') as f:
        lines = f.readlines()
    data = np.zeros((len(lines), char_num_of_sentence, SIZE_OF_VOCAB))
    for line_id, line in enumerate(lines):
        line = line.lower()[0:char_num_of_sentence-2]
        line += ' ' * (char_num_of_sentence-2 - len(line))
        data[line_id] = convert_to_one_hot(line)

    # Convert to 1-hot coding
    return data
