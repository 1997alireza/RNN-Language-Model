import time
from LSTM_tensorflow.tools import *
import tensorflow as tf
import os


class LSTM_NN:

    def __init__(self, session, check_point_dir, hidden_size=256, num_layers=2, lr=0.003,
                 scope_name="RNN"):
        self.scope = scope_name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.session = session
        self.lr = tf.constant(lr)
        self.check_point_dir = check_point_dir

        # Defining the computational graph

        self.lstm_last_state = np.zeros(
            (self.num_layers * 2 * self.hidden_size,)
            # num_layer * 2 (one for h and one for c) * hidden_size
        )
        with tf.variable_scope(self.scope):
            self.x_batch = tf.placeholder(
                tf.float32,
                shape=(None, CHAR_NUM_OF_SENTENCE, SIZE_OF_VOCAB),
                name="input"
            )
            self.lstm_init_value = tf.placeholder(
                tf.float32,
                shape=(None, self.num_layers * 2 * self.hidden_size),
                # None -> number of sentences in this batch
                name="lstm_init_value"
            )

            # LSTM
            self.lstm_cells = [
                tf.contrib.rnn.BasicLSTMCell(
                    self.hidden_size,
                    forget_bias=1.0,
                    state_is_tuple=False
                ) for _ in range(self.num_layers)
            ]
            self.lstm = tf.contrib.rnn.MultiRNNCell(
                self.lstm_cells,
                state_is_tuple=False
            )
            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(
                self.lstm,
                self.x_batch,
                initial_state=self.lstm_init_value,
                dtype=tf.float32
            )
            # fc layer at the end
            self.W = tf.Variable(
                tf.random_normal(
                    (self.hidden_size, SIZE_OF_VOCAB),
                    stddev=0.01
                )
            )
            self.B = tf.Variable(
                tf.random_normal(
                    (SIZE_OF_VOCAB,), stddev=0.01
                    # size : SIZE_OF_VOCAB (1d)
                )
            )
            outputs_reshaped = tf.reshape(outputs, [-1, self.hidden_size])
            network_output = tf.matmul(
                outputs_reshaped,
                self.W
            ) + self.B

            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(
                tf.nn.softmax(network_output),
                (batch_time_shape[0], batch_time_shape[1], SIZE_OF_VOCAB)
            )

            self.y_batch = tf.placeholder(
                tf.float32,
                (None, None, SIZE_OF_VOCAB)
            )
            y_batch_long = tf.reshape(self.y_batch, [-1, SIZE_OF_VOCAB])
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=network_output,
                    labels=y_batch_long
                )
            )
            # minimizing the error
            self.train_op = tf.train.RMSPropOptimizer(
                self.lr,
                0.9
            ).minimize(self.cost)

    def train_on_batch(self, x_batch, y_batch):
        """

        :param x_batch: size=(batch_size, char_num_of_sentence, SIZE_OF_VOCAB)
        :param y_batch: size=(batch_size, char_num_of_sentence, SIZE_OF_VOCAB)
        :return:
        """
        init_value = np.zeros(
            (x_batch.shape[0], self.num_layers * 2 * self.hidden_size)
        )
        cost, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.x_batch: x_batch,
                self.y_batch: y_batch,
                self.lstm_init_value: init_value
            }
        )
        return cost

    def run_step(self, x, init_zero_state=True):
        if init_zero_state:
            init_value = np.zeros((self.num_layers * 2 * self.hidden_size,))
        else:
            init_value = self.lstm_last_state
        out, next_lstm_state = self.session.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict={
                self.x_batch: [x],  # todo: chotor mishe? mage 3d nbud?
                self.lstm_init_value: [init_value]
            }
        )
        self.lstm_last_state = next_lstm_state[0]
        print("OUT:", out)
        #  todo: check whats the out?
        return out[0][0]


def load_model(check_point_dir):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    net = LSTM_NN(
        check_point_dir=check_point_dir,
        session=sess,
        scope_name="char_rnn_network"
    )
    check_point = check_point_dir + '\model.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if os.path.exists(check_point):
        saver.restore(sess, check_point)
    return net


def train(model, data, batch_size=64, time_steps=200, num_train_batches=20000):
    # data, vocab = load_data(input_file)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = model.session
    print("y1")
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    check_point = model.check_point_dir + '\model.ckpt'

    last_time = time.time()

    np_start_token = [0.0] * SIZE_OF_VOCAB
    np_start_token[START] = 1.0
    np_end_token = [0.0] * SIZE_OF_VOCAB
    np_end_token[END] = 1.0

    batch_size = len(data)
    print(batch_size)

    print("y2")
    for i in range(num_train_batches):
        x_batch = np.zeros((batch_size, CHAR_NUM_OF_SENTENCE, SIZE_OF_VOCAB))
        y_batch = np.zeros((batch_size, CHAR_NUM_OF_SENTENCE, SIZE_OF_VOCAB))

        print("ii", i)
        for batch_id in range(batch_size):
            print(np.shape(np_start_token))
            print(np.shape(np_end_token))
            print(np.shape(data))
            print(np.shape(data[batch_id]))
            x_batch[batch_id] = np.append([np_start_token], data[batch_id], axis=0)
            y_batch[batch_id] = np.append(data[batch_id], [np_end_token], axis=0)

        print("  x")
        batch_cost = model.train_on_batch(x_batch, y_batch)
        print("  y")

        if (i % 100) == 0:
            new_time = time.time()
            diff = new_time - last_time
            last_time = new_time
            print("batch: {}  loss: {}  speed: {} batches / s".format(
                i, batch_cost, 100 / diff
            ))

            saver.save(sess, check_point)


# TODO: change state to be a tuple
# todo: don't find the 1. find the max

def predict(prefix, model, generate_len):#TODO
    """prefix = prefix.lower()
    for i in range(len(prefix)):
        out = model.run_step(convert_to_one_hot_old(prefix[i], vocab)) # todo: lidi

    print("Sentence:")
    gen_str = prefix
    for i in range(generate_len):
        element = np.random.choice(range(len(vocab)), p=out)
        gen_str += vocab[element]
        out = model.run_step(convert_to_one_hot_old(vocab[element], vocab), False)

    print(gen_str)"""


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    saver_directory = './saved'

    CHAR_NUM_OF_SENTENCE = 100
    data = load_data('../datasets/shakespeare -all.txt', CHAR_NUM_OF_SENTENCE)

    test = False
    if not test:
        model = LSTM_NN(sess, saver_directory)
        train(model, data)
    else:
        model = load_model(saver_directory)
        predict('I am ', model, 500)
