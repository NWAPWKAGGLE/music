import tensorflow as tf
from tqdm import tqdm
import numpy as np
import midi_manipulation as mm
class LSTM:

    def __init__(self, model_name, song_dir, num_features, layer_units, batch_size, load_from_saved=False, max_songs=None, learning_rate = .05):

        #Set Hyperparams
        self.model_name = model_name
        self.song_dir = song_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.load_from_saved = load_from_saved
        self.num_features = num_features
        self.layer_units = layer_units
        self.max_songs = max_songs


        self.sess = None
        self.saver = None

        #build model - this part should probably be abstracted somehow, good ideas on how to do that possibly here https://danijar.com/structuring-your-tensorflow-models/
        self.x = tf.placeholder(tf.float32, (None, None, self.num_features), name='x')
        self.y = tf.placeholder(tf.float32, (None, None, self.num_features), name='y')

        self.G_W1 = tf.Variable(tf.truncated_normal([self.layer_units, self.num_features], stddev=.1), name='G_W1')
        self.G_b1 = tf.Variable(tf.truncated_normal([self.num_features], stddev=.1), name='G_b1')

        self.G_vars = [self.G_W1, self.G_b1]

        self.D_W1 = tf.Variable(tf.truncated_normal([self.layer_units, 1], stddev=.1), name='D_W1')
        self.D_b1 = tf.Variable(tf.truncated_normal([1], stddev=.1), name='D_b1')

        self.D_vars = [self.D_W1, self.D_b1]

        self.seq_len = tf.placeholder(tf.int32, (None,), name='seq_lens')

        with tf.variable_scope('generator_lstm_layer{0}'.format(1)):
            self.generator_lstm_cell = tf.contrib.rnn.LSTMCell(layer_units)

        self.states = None

        with tf.variable_scope('discriminator_lstm_layer{0}'.format(1)):
            self.discriminator_lstm_cell = tf.contrib.rnn.LSTMCell(layer_units)


        self.G_sample = self.generator(self.x)

        self.D_real = self.discriminator(self.x)
        self.D_fake = self.discriminator(self.G_sample)

        self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1. - self.D_fake), name='D_loss')
        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake), name='G_loss')

        self.D_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='D_optimizer').minimize(self.D_loss,
                                                                                                          var_list=self.D_vars)
        self.G_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='G_optimizer').minimize(self.G_loss,
                                                                                                          var_list=self.G_vars)
    def start_sess(self):
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if self.load_from_saved:
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./model_saves/{}/'.format(self.model_name, self.model_name)))
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def end_sess(self):
        self.saver.save(self.sess, './model_saves/{}/{}'.format(self.model_name, self.model_name))
        self.sess.close()

    def discriminator(self, inputs):
        with tf.variable_scope('discriminator_lstm_layer{0}'.format(1)):
            discriminator_outputs, states = tf.nn.dynamic_rnn(self.discriminator_lstm_cell, inputs, dtype=tf.float32,
                                                              sequence_length=self.seq_len)
        discriminator_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, self.D_W1) + self.D_b1),
                                          discriminator_outputs, name='D_')
        return discriminator_outputs

    def generator(self, inputs, reuse_states = False):

        with tf.variable_scope('generator_lstm_layer{0}'.format(1)):
            #reuse states if necessary
            if reuse_states:
                states = self.states
            else:
                states = None
            generator_outputs, states = tf.nn.dynamic_rnn(self.generator_lstm_cell, inputs, dtype=tf.float32,
                                                          sequence_length=self.seq_len, initial_state=states)
            if reuse_states:
                self.states = states

        generator_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, self.G_W1) + self.G_b1), generator_outputs,
                                      name='G_')
        return generator_outputs

    def generateSequence(self, starter, numsteps):
        #this needs to be fixed to use all the starter values
        output = [starter[0]]

        for i in tqdm(range(numsteps)):
            #runs the generate with reusing states
            oneoutput = self.sess.run(self.generator(self.x, reuse_states=True), feed_dict={self.x: np.expand_dims(output[-1], 0), self.seq_len: [1]})[-1]
            output.append(oneoutput)
        #set states to None in case generate Sequence is used
        self.states = None
        return np.round(output).astype(int)

    def generate_midi_from_sequences(self, sequence, dir_path):
        for i in range(len(sequence)):
            mm.noteStateMatrixToMidi(sequence[i], dir_path+'generated_chord_{}'.format(i))

    def trainAdversarially(self, training_input, training_expected, epochs, report_interval = 10, seqlens = None):
        iter_ = tqdm(range(epochs), desc="{0}.learn".format(self.model_name))

        for i in iter_:

            self.sess.run('G_optimizer', feed_dict={self.x: training_input, self.y: training_expected, self.seq_len: seqlens})

            self.sess.run('D_optimizer', feed_dict={self.x: training_input, self.y: training_expected, self.seq_len: seqlens})

            if i % report_interval == 0:
                # save
                print('G Error {}'.format(
                    self.sess.run(self.G_loss,feed_dict={self.x: training_input, self.y: training_expected, self.seq_len: seqlens})))
                print('D Error {}'.format(
                    self.sess.run(self.D_loss, feed_dict={self.x: training_input, self.y: training_expected, self.seq_len: seqlens})))
