import numpy as np


# main characters: 0-127 = 0-127
# persian characters: (char code) 1560-1751 <-> (id) 128-319
# <start> = 320, <end> = 321, <unk> = 322
START = 320
END = 321
UNK = 322
SIZE_OF_VOCAB = 323


def map_char_to_id(c):
    """

    :return: a number between 0 to 319 or 322
    """
    code = ord(c)
    if code < 128:
        return code
    if 1560 <= code <= 1751:
        return code - 1432
    return UNK


def map_id_to_char(code):
    """

    :param code: a number between 0 to 322
    """
    if code == START: return '<START>'
    if code == END: return '<END>'
    if code == UNK: return '<UNK>'
    if code < 128: return chr(code)
    if 128 <= code <= 319: return chr(code + 1432)
    return '<?>'


def convert_to_one_hot_old(data_, vocab):  # todo: delete this function
    data = np.zeros((len(data_), len(vocab)))
    for cnt, s in enumerate(data_):
        v = [0.0] * len(vocab)
        print(s)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
    return data


def convert_to_one_hot(sentence, char_num):
    """

    :param char_num: char_num_of_sentence-1
    :return: it's the one-hot array of the sentence without start and end token
    """
    s_vector = np.zeros((char_num, SIZE_OF_VOCAB))
    for char_id, char_c in enumerate(sentence):
        v = [0.0] * SIZE_OF_VOCAB
        v[map_char_to_id(char_c)] = 1.0
        s_vector[char_id] = v

    return s_vector


def load_data(input, char_num_of_sentence):
    """

    :param input: data file address
    :param char_num_of_sentence: fix char size of any sentences
    :return: a numpy array with the size (len(lines), char_num_of_sentence-1, SIZE_OF_VOCAB)
    """
    # Load the data
    with open(input, 'r') as f:
        lines = f.readlines()
    data = np.zeros((len(lines), char_num_of_sentence-1, SIZE_OF_VOCAB))
    for line_id, line in enumerate(lines):
        line = line.lower()[0:char_num_of_sentence-1]
        line += ' ' * (char_num_of_sentence-1 - len(line))

        # Convert to 1-hot coding
        data[line_id] = convert_to_one_hot(line, char_num_of_sentence-1)
    return data
