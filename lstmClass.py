import numpy as np
import tensorflow as tf
from tqdm import tqdm

import midi_manipulation as mm


class LSTM:
    def __init__(self, model_name, num_features, layer_units, batch_size, learning_rate=.05, num_layers=3):
        """
        :param model_name: (path, string) the name of the model, for saving and loading
        :param num_features: (int) the number of features the model uses (156 in this case)
        :param layer_units: (int) the number of units in the lstm layer(s)
        :param batch_size: (int) the size of each training batch (num_songs)
        :param load_from_saved: (bool), whether or not to load a model back from a save
        :param learning_rate: (int) the learning rate for the model
        """

        # Set Hyperparams
        self.model_name = model_name

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_features = num_features
        self.layer_units = layer_units

        self.sess = None
        self.saver = None

        # build model - this part should probably be abstracted somehow,
        # good ideas on how to do that possibly here https://danijar.com/structuring-your-tensorflow-models/
        self.x = tf.placeholder(tf.float32, (None, None, self.num_features), name='x')
        self.y = tf.placeholder(tf.float32, (None, None, self.num_features), name='y')

        self.seq_len = tf.placeholder(tf.int32, (None,), name='seq_lens')

        with tf.variable_scope('generator') as scope:
            self.G_W1 = tf.Variable(tf.truncated_normal([self.layer_units, self.num_features], stddev=.1), name='G_W1')
            self.G_b1 = tf.Variable(tf.truncated_normal([self.num_features], stddev=.1), name='G_b1')

            self.generator_lstm_cell = self.lstm_cell_construct(layer_units, num_layers)

            self.G_vars = scope.trainable_variables()

        with tf.variable_scope('discriminator') as scope:
            self.D_W1 = tf.Variable(tf.truncated_normal([self.layer_units * 2, 1], stddev=.1), name='D_W1')
            self.D_b1 = tf.Variable(tf.truncated_normal([1], stddev=.1), name='D_b1')

            with tf.variable_scope('fw'):
                self.discriminator_lstm_cell_fw = self.lstm_cell_construct(layer_units, num_layers)
            with tf.variable_scope('bw'):
                self.discriminator_lstm_cell_bw = self.lstm_cell_construct(layer_units, num_layers)

            self.D_vars = scope.trainable_variables()

        self.states = None


        self.G_sample = self.generator(self.x)

        self.D_real = self.discriminator(self.x)
        self.D_fake = self.discriminator(self.G_sample)

        self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1. - self.D_fake), name='D_loss')
        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake), name='G_loss')

        self.D_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='D_optimizer').minimize(
            self.D_loss,
            var_list=self.D_vars)
        self.G_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='G_optimizer').minimize(
            self.G_loss,
            var_list=self.G_vars)
        self.cost = tf.identity(tf.losses.softmax_cross_entropy(self.y, logits=self.G_sample), name='cost')
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, name='optimizer').minimize(
            self.cost)

    def lstm_cell_construct(self, layer_units, num_layers):
        cell_list = []
        for i in range(num_layers):
            with tf.variable_scope('layer_{0}'.format(i)):
                cell_list.append(tf.contrib.rnn.LSTMCell(layer_units))
        return tf.contrib.rnn.MultiRNNCell(cell_list)

    def start_sess(self, load_from_saved=False):
        """
        starts a tensorflow session to run model functions in, loads in a save model if specified
        :return: None
        """
        self.saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=0.5)
        self.sess = tf.Session()

        if load_from_saved:
            self.saver.restore(self.sess,
                               tf.train.latest_checkpoint('./model_saves/{}/'.format(self.model_name, self.model_name)))
            print('loaded from save')
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)
            print('new model')

    def end_sess(self):
        """
        ends the tensorflow sess, saves the model
        :return: None
        """
        dir = self.saver.save(self.sess, './model_saves/{}/{}'.format(self.model_name, self.model_name))
        print(dir)
        self.sess.close()

    def discriminator(self, inputs):
        """

        :param inputs: (tf.Tensor, shape: (Batch_Size, Time_Steps, Num_Features)) the inputs to the discriminator lstm
        :return: (tf.Tensor, (Batch_Size, Time_Steps, 1)) the outputs of the discriminator lstm
        (single values denoting real or fake samples)
        """

        with tf.variable_scope('discriminator_lstm_layer{0}'.format(1)):
            #discriminator_outputs, states = tf.nn.dynamic_rnn(self.discriminator_lstm_cell, inputs, dtype=tf.float32,
            #                                                  sequence_length=self.seq_len)
            discriminator_outputs, states = tf.nn.bidirectional_dynamic_rnn(self.discriminator_lstm_cell_fw,
                self.discriminator_lstm_cell_bw, inputs, dtype=tf.float32, sequence_length = self.seq_len)
            discriminator_outputs_fw, discriminator_outputs_bw = discriminator_outputs
            discriminator_outputs = tf.concat([discriminator_outputs_fw, discriminator_outputs_bw], 2)
        discriminator_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, self.D_W1) + self.D_b1),
                                          discriminator_outputs, name='D_')
        return discriminator_outputs

    def generator(self, inputs, reuse_states=False, time_major=False):
        """

        :param inputs: (tf.Tensor, shape: (Batch_Size, Time_Steps, Num_Features)) inputs into the generator lstm
        :param reuse_states: (Bool) whether to reuse previous lstm states, for use when generating long sequences recursively. default
        :param time_major: (Bool) whether to set time_major to true for the lstm cell
        :return: (tf.Tensor, shape: (Batch_Size, Time_Steps, Num_Features)) outputs from the generator lstm
        """
        with tf.variable_scope('generator_lstm_layer{0}'.format(1)):
            # reuse states if necessary
            if reuse_states:
                states = self.states
            else:
                states = None
            generator_outputs, states = tf.nn.dynamic_rnn(self.generator_lstm_cell, inputs, dtype=tf.float32,
                                                          sequence_length=self.seq_len, initial_state=states,
                                                          time_major=time_major)
            if reuse_states:
                self.states = states

        generator_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, self.G_W1) + self.G_b1),
                                      generator_outputs,
                                      name='G_')
        return generator_outputs

    def generateSequence(self, starter, numsteps):
        """

        :param starter: (np.ndarray) starter sequence to use for recursive generation
        :param numsteps: (int) the number of timesteps to generate
        :return: (np.ndarray, shape: (num_songs, numsteps, num_features)) an array of songs
        """
        # this needs to be fixed to use all the starter values
        output = starter
        self.states = None
        gen = self.generator(self.x, reuse_states=True, time_major=True)


        for i in tqdm(range(numsteps)):
            # runs the generate with reusing states
            oneoutput = self.sess.run(gen,
                                      feed_dict={self.x: np.expand_dims(output[-1], 0),
                                                 self.seq_len: [1 in range(len(starter))]})[-1]
            np.append(output, oneoutput)
        # set states to None in case generate Sequence is used
        self.states = None
        return np.around(np.transpose(output, (1, 0, 2))).astype(int)

    def generate_midi_from_sequences(self, sequence, dir_path):
        """

        :param sequence: (np.ndarray, shape: (num_songs, numsteps, num_features)) an array of songs,
        like is outputed from self.generateSequence()
        :param dir_path: (string, path) the directory to save the songs to
        :return: None
        """
        for i in range(len(sequence)):
            mm.noteStateMatrixToMidi(sequence[i], dir_path + 'generated_chord_{}'.format(i))

    def trainAdversarially(self, training_input, training_expected, epochs, report_interval=10, seqlens=None):
        """

        :param training_input: 
        :param training_expected:
        :param epochs:
        :param report_interval:
        :param seqlens:
        :return:
        """
        iter_ = tqdm(range(epochs), desc="{0}.learn".format(self.model_name))

        for i in iter_:

            self.sess.run('G_optimizer',
                          feed_dict={self.x: training_input, self.y: training_expected, self.seq_len: seqlens})

            self.sess.run('D_optimizer',
                          feed_dict={self.x: training_input, self.y: training_expected, self.seq_len: seqlens})

            if i % report_interval == 0:
                self.saver.save(self.sess, self.model_name, global_step=i)
                print('G Error {}'.format(
                    self.sess.run(self.G_loss, feed_dict={self.x: training_input, self.y: training_expected,
                                                          self.seq_len: seqlens})))
                print('D Error {}'.format(
                    self.sess.run(self.D_loss, feed_dict={self.x: training_input, self.y: training_expected,
                                                          self.seq_len: seqlens})))

    def trainLSTM(self, training_input, training_expected, epochs, report_interval=10, seqlens=None):
        iter_ = tqdm(range(epochs), desc="{0}.learn".format(self.model_name))

        for i in iter_:
            self.sess.run('optimizer',
                          feed_dict={self.x: training_input, self.y: training_expected, self.seq_len: seqlens})

            if i % report_interval == 0:
                print('Error {}'.format(
                    self.sess.run(self.cost,
                                  feed_dict={self.x: training_input, self.y: training_expected, self.seq_len: seqlens})
                ))
