import numpy as np
# from tools import read_data
from RNN_numpy.tools import read_data

HIDDEN_LAYER_SIZE = 100
CHUNK_SIZE = 25  # number of steps on input to unroll the RNN
LEARNING_RATE = 1e-1
THRESHOLD = 5
LOG_ITER = 1000

w_xh, w_hh, w_hy, b_h, b_y = [None]*5


def initialize_rnn():
    """
    initialize the weights and the biases of the network
    """
    global w_xh, w_hh, w_hy, b_h, b_y
    w_xh = np.random.randn(HIDDEN_LAYER_SIZE, vocab_size) * 0.01
    w_hh = np.random.randn(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) * 0.01
    w_hy = np.random.randn(vocab_size, HIDDEN_LAYER_SIZE) * 0.01
    b_h = np.zeros((HIDDEN_LAYER_SIZE, 1))
    b_y = np.zeros((vocab_size, 1))


def predict(h_state, prefix_ids, predict_n):
    """

    :param h_state: initial value for hidden layer
    :param prefix_ids: an array containing inputs
    :param predict_n: predict outputs on predict_n iterations
    :return: predicted output string
    """

    # predicted_on_prefix = []
    h = h_state
    x = np.zeros((vocab_size, 1))

    for idx in prefix_ids:
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        h = np.tanh(np.dot(w_xh, x) + np.dot(w_hh, h) + b_h)
        # y = np.dot(w_hy, h) + b_y
        # p = np.exp(y) / np.sum(np.exp(y))
        # o_idx = np.random.choice(range(vocab_size), p=p.ravel())
        # predicted_on_prefix.append(o_idx)

    # predicted_on_prefix_str = [char_at[o_idx] for o_idx in predicted_on_prefix]
    # print('predicted_on_prefix_str: ', predicted_on_prefix_str)

    predicted = []
    for t in range(predict_n):
        h = np.tanh(np.dot(w_xh, x) + np.dot(w_hh, h) + b_h)
        y = np.dot(w_hy, h) + b_y
        p = np.exp(y) / np.sum(np.exp(y))
        o_idx = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[o_idx] = 1
        predicted.append(o_idx)

    predicted_str = [char_at[o_idx] for o_idx in predicted]
    # print('predicted_str: ', predicted_str)

    return predicted_str


def gradients_and_loss(inputs, targets, h_state):
    """

    :param inputs: input list for the RNN
    :param targets: target list for the RNN
    :param h_state: initial value for the hidden state on this chunk
    :return: the loss, gradients and last hidden state
    """
    x, y, p = [[None] * len(inputs)] * 3
    # p : probabilities for next chars using softmax
    h = {-1: np.copy(h_state)}
    loss = 0

    # we didn't make any changes on the neural network parameters,
    # we just pass through the hidden layers to make the outputs according to the inputs

    # forward pass
    for t in range(len(inputs)):
        x[t] = np.zeros((vocab_size, 1))
        x[t][inputs[t]] = 1
        h[t] = np.tanh(np.dot(w_xh, x[t]) + np.dot(w_hh, h[t-1]) + b_h)
        y[t] = np.dot(w_hy, h[t]) + b_y
        p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))
        loss += -np.log(p[t][targets[t], 0])  # cross-entropy loss

    # backward pass (compute gradients)
    d_w_xh, d_w_hh, d_w_hy, d_b_h, d_b_y = [np.zeros_like(param) for param in [w_xh, w_hh, w_hy, b_h, b_y]]
    d_h_next = np.zeros_like(h[0])
    for t in reversed(range(len(inputs))):
        d_y = np.copy(p[t])
        d_y[targets[t]] -= 1
        # back-propagation into y. see http://cs231n.github.io/neural-networks-case-study/#grad for more details
        # Todo: understand these mathematics!
        d_w_hy += np.dot(d_y, h[t].T)
        d_b_y += d_y
        d_h = np.dot(w_hy.T, d_y) + d_h_next  # back-propagation into h
        d_h_raw = (1 - h[t] ** 2) * d_h  # back-propagation through tanh non-linearly
        d_b_h += d_h_raw
        d_w_xh += np.dot(d_h_raw, x[t].T)
        d_w_hh += np.dot(d_h_raw, h[t-1].T)
        d_h_next = np.dot(w_hh.T, d_h_raw)

    for d_param in [d_w_xh, d_w_hh, d_w_hy, d_b_h, d_b_y]:
        np.clip(d_param, -THRESHOLD, THRESHOLD, out=d_param)  # to mitigate exploding gradients

    return loss, d_w_xh, d_w_hh, d_w_hy, d_b_h, d_b_y, h[len(inputs)-1]


def execute_rnn():
    iteration, whole_input_iteration, pointer = 0, 0, 0

    sg_w_xh = np.zeros_like(w_xh)
    sg_w_hh = np.zeros_like(w_hh)
    sg_w_hy = np.zeros_like(w_hy)
    sg_b_h  = np.zeros_like(b_h)
    sg_b_y  = np.zeros_like(b_y)
    # sum of squares of gradients, used to decreasing the step lengths by AdaGrad algorithm

    smooth_loss = np.log(vocab_size) * CHUNK_SIZE  # initial value

    h_state = np.zeros((HIDDEN_LAYER_SIZE, 1))

    while True:
        if pointer + 1 >= data_size:
            h_state = np.zeros((HIDDEN_LAYER_SIZE, 1))
            pointer = 0
            whole_input_iteration += 1
        elif pointer + CHUNK_SIZE >= data_size:
            pointer = data_size - CHUNK_SIZE - 1

        inputs  = [id_of[c] for c in data[pointer: pointer + CHUNK_SIZE]]          # inputs  : p ... p + chunk size - 1
        targets = [id_of[c] for c in data[pointer + 1: pointer + CHUNK_SIZE + 1]]  # targets : p + 1 ... p + chunk size

        if iteration % LOG_ITER == 0:
            print(prefix, ''.join(predict(h_state, prefix_ids, predict_char_nums)))

        loss, d_w_xh, d_w_hh, d_w_hy, d_b_h, d_b_y, h_state = gradients_and_loss(inputs, targets, h_state)
        smooth_loss = smooth_loss * .999 + loss * .001

        if iteration % LOG_ITER == 0:
            print('---------------------------------------------\n',
                  'whole iter %d, iter %d, loss: %f' % (whole_input_iteration, iteration, smooth_loss),
                  '\n---------------------------------------------\n\n')

        for param, d_param, sg_param in zip([w_xh, w_hh, w_hy, b_h, b_y],
                                            [d_w_xh, d_w_hh, d_w_hy, d_b_h, d_b_y],
                                            [sg_w_xh, sg_w_hh, sg_w_hy, sg_b_h, sg_b_y]):
            sg_param += d_param ** 2
            param += -LEARNING_RATE * d_param / np.sqrt(sg_param + 1e-8)
            # AdaGrad algorithm

        pointer += CHUNK_SIZE
        iteration += 1


if __name__ == '__main__':
    data, char_vocab = read_data('datasets/sample.txt')
    data_size, vocab_size = len(data), len(char_vocab)
    id_of = {c: i for i, c in enumerate(char_vocab)}
    char_at = {i: c for i, c in enumerate(char_vocab)}
    # TODO: add unknown to vocab

    prefix = 'In Warwickshire'
    prefix_ids = [id_of[c] for c in prefix]
    predict_char_nums = 100

    initialize_rnn()
    execute_rnn()
