import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
import os
from datetime import datetime

np.set_printoptions(threshold=np.nan)

def split_list(l, n):
    list = []
    for j in range(0, len(l), n):
        if (j+n < len(l)):
            list.append(np.array(l[j:j+n]))
    return list

def sample(probs):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

import midi_manipulation as mm

class LSTM:
    def __init__(self, model_name, num_features, layer_units, batch_size, n_hidden_RBM, gibbs_sample_steps = 1, learning_rate_RBM = .005, learning_rate=.05, num_layers=2):
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

        self.n_visible_RBM = num_features

        # size of the hidden layer
        self.n_hidden_RBM = n_hidden_RBM

        # number of training examples to send through at a time

        # number of steps to take when doing a gibbs sample
        self.gibbs_sample_steps = gibbs_sample_steps

        # the learning rate
        self.lr_RBM = tf.constant(learning_rate_RBM, tf.float32)

        self.sess = None
        self.saver = None
        self.writer = None
        # build model - this part should probably be abstracted somehow,
        # good ideas on how to do that possibly here https://danijar.com/structuring-your-tensorflow-models/
        self.x = tf.placeholder(tf.float32, (None, None, self.num_features), name='x')
        self.y = tf.placeholder(tf.float32, (None, None, self.num_features), name='y')

        self.seq_len = tf.placeholder(tf.int32, (None,), name='seq_lens')

        with tf.variable_scope('generator') as scope:


            self.G_vars = []

            # size of visible layer

            self.x_RBM = tf.placeholder(tf.float32, [None, self.n_visible_RBM], name="x_RBM")
            self.W = tf.Variable(tf.random_normal([self.n_visible_RBM, self.n_hidden_RBM], 0.01), name="W")

            # bias vector for hidden layer
            self.bh = tf.Variable(tf.zeros([1, self.n_hidden_RBM], tf.float32, name="bh"))
            # bias vector for visible layer
            self.bv = tf.Variable(tf.zeros([1, self.n_visible_RBM], tf.float32, name="bv"))


            #### Generative Algorithm
            # sample of x
            self.x_sample = self.gibbs_sample(1)
            # sample of hidden nodes, from original inputs
            self.h = sample(tf.sigmoid(tf.matmul(self.x_RBM, self.W) + self.bh))
            # sample of hidden nodes, from sampled reconstructed inputs
            self.h_sample = sample(tf.sigmoid(tf.matmul(self.x_sample, self.W) + self.bh))

            # update weights and biases based on the differences between created samples and original values

            self.size_bt = tf.cast(tf.shape(self.x_RBM)[0], tf.float32)
            print(self.size_bt)
            self.W_adder = tf.multiply(self.lr_RBM / self.size_bt, tf.subtract(tf.matmul(tf.transpose(self.x_RBM), self.h),
                                                                           tf.matmul(tf.transpose(self.x_sample),
                                                                                     self.h_sample)))
            self.bv_adder = tf.multiply(self.lr_RBM / self.size_bt,
                                        tf.reduce_sum(tf.subtract(self.x_RBM, self.x_sample), 0, True))
            self.bh_adder = tf.multiply(self.lr_RBM / self.size_bt,
                                        tf.reduce_sum(tf.subtract(self.h, self.h_sample), 0, True))
            # when we do sess.run(upt), TF will do all 3 update steps
            self.updt = [self.W.assign_add(self.W_adder), self.bv.assign_add(self.bv_adder),
                         self.bh.assign_add(self.bh_adder)]

            self.G_W1 = tf.Variable(tf.truncated_normal([self.layer_units, self.num_features], stddev=.1), name='G_W1')
            self.G_b1 = tf.Variable(tf.truncated_normal([self.num_features], stddev=.1), name='G_b1')

            self.generator_lstm_cell, gen_vars = self.lstm_cell_construct(layer_units, num_layers)

            self.G_vars.extend(gen_vars)
            self.G_vars.extend(scope.trainable_variables())

        with tf.variable_scope('discriminator') as scope:
            self.D_vars = []

            self.D_W1 = tf.Variable(tf.truncated_normal([self.layer_units, 1], stddev=.1), name='D_W1')
            self.D_b1 = tf.Variable(tf.truncated_normal([1], stddev=.1), name='D_b1')

            with tf.variable_scope('fw') as subscope:
                self.discriminator_lstm_cell_fw, fw_vars = self.lstm_cell_construct(layer_units, num_layers)
            with tf.variable_scope('bw') as subscope:
                self.discriminator_lstm_cell_bw, bw_vars = self.lstm_cell_construct(layer_units, num_layers)

            self.D_vars.extend(fw_vars)
            self.D_vars.extend(bw_vars)
            self.D_vars.extend(scope.trainable_variables())

        self.states = None

        self.G_sample, g_vars = self.generator(self.x)
        
        self.G_vars.extend(g_vars)

        self.D_real, _ = self.discriminator(self.x) # returns same d_vars; unnecessary to use this return value here
        self.D_fake, d_vars = self.discriminator(self.G_sample)

        self.real_count = tf.reduce_mean(self.D_real)
        self.fake_count = tf.reduce_mean(self.D_fake)

        self.D_vars.extend(d_vars)

        self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1. - self.D_fake), name='D_loss')
        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake), name='G_loss')

        print(self.G_vars)
        print(self.D_vars)

        self.D_optimizer = tf.train.GradientDescentOptimizer(learning_rate=.05, name='D_optimizer').minimize(
            self.D_loss,
            var_list=self.D_vars)
        self.G_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='G_optimizer').minimize(
            self.G_loss,
            var_list=self.G_vars)

    def gibbs_sample(self, k):
        # Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
        def gibbs_step(count, k, xk):
            # Runs a single gibbs step. The visible values are initialized to xk
            hk = sample(
                tf.sigmoid(tf.matmul(xk, self.W) + self.bh))  # Propagate the visible values to sample the hidden values
            xk = sample(tf.sigmoid(
                tf.matmul(hk,
                          tf.transpose(self.W)) + self.bv))  # Propagate the hidden values to sample the visible values
            return count + 1, k, xk

        # Run gibbs steps for k iterations
        ct = tf.constant(0)  # counter
        [_, _, x_sample] = tf.while_loop(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), self.x_RBM], None, 1, False)
        # This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
        # optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
        x_sample = tf.stop_gradient(x_sample)
        return x_sample

    def train(self, num_epochs, training_set):

        # run through training data num_epoch times
        for epoch in tqdm(range(num_epochs)):
            self.train_step(training_set)

    def train_step(self, training_set):
        for i in range(1, len(training_set), self.batch_size):
            tr_x = training_set[i:i + self.batch_size]
            self.sess.run(self.updt, feed_dict={self.x_RBM: tr_x})

    def lstm_cell_construct(self, layer_units, num_layers):
        cell_list = []
        var_list = []
        for i in range(num_layers):
            with tf.variable_scope('layer_{0}'.format(i)) as scope:
                cell = tf.contrib.rnn.LSTMCell(layer_units)
                cell_list.append(tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=.5, input_keep_prob=.9))
                var_list.extend(scope.trainable_variables())
        return tf.contrib.rnn.MultiRNNCell(cell_list), var_list

    def start_sess(self, load_from_saved=False):
        """
        starts a tensorflow session to run model functions in, loads in a save model if specified
        :return: None
        """
        self.saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=0.5)
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter("output", self.sess.graph)
        if load_from_saved:
            self.saver.restore(self.sess,
                               tf.train.latest_checkpoint('./model_saves/{}/'.format(self.model_name)))
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
        self.writer.close()
        dir = self.saver.save(self.sess, './model_saves/{}/{}_{}'.format(self.model_name, self.model_name, 'end_sess'))
        self.sess.close()

    def discriminator(self, inputs):
        """

        :param inputs: (tf.Tensor, shape: (Batch_Size, Time_Steps, Num_Features)) the inputs to the discriminator lstm
        :return: (tf.Tensor, (Batch_Size, Time_Steps, 1)) the outputs of the discriminator lstm
        (single values denoting real or fake samples)
        """

        with tf.variable_scope('discriminator_lstm_layer{0}'.format(1)) as scope:
            #discriminator_outputs, states = tf.nn.dynamic_rnn(self.discriminator_lstm_cell, inputs, dtype=tf.float32,
            #                                                  sequence_length=self.seq_len)
            discriminator_outputs, states = tf.nn.bidirectional_dynamic_rnn(self.discriminator_lstm_cell_fw,
                self.discriminator_lstm_cell_bw, inputs, dtype=tf.float32)
            discriminator_outputs_fw, discriminator_outputs_bw = discriminator_outputs
            discriminator_outputs = tf.concat([discriminator_outputs_fw, discriminator_outputs_bw], axis=1)
            d_vars = scope.trainable_variables()
        discriminator_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, self.D_W1) + self.D_b1),
                                          discriminator_outputs, name='D_')
        return discriminator_outputs, d_vars

    def generator(self, inputs):
        """
        :param inputs: (tf.Tensor, shape: (Batch_Size, Time_Steps, Num_Features)) inputs into the generator lstm
        :param reuse_states: (Bool) whether to reuse previous lstm states, for use when generating long sequences recursively. default
        :param time_major: (Bool) whether to set time_major to true for the lstm cell
        :return: (tf.Tensor, shape: (Batch_Size, Time_Steps, Num_Features)) outputs from the generator lstm
        """

        with tf.variable_scope('generator_lstm_layer{0}'.format(1)) as scope:
            # reuse states if necessary

            generator_outputs, states = tf.nn.dynamic_rnn(self.generator_lstm_cell, inputs, dtype=tf.float32,
                                                          sequence_length=self.seq_len)
            g_vars = scope.trainable_variables()
        generator_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, self.G_W1) + self.G_b1),
                                      generator_outputs,
                                      name='G_')

        return generator_outputs, g_vars

    def generate_sequence(self, num_songs, num_steps):
        """

        :param starter: (np.ndarray) starter sequence to use for recursive generation
        :param numsteps: (int) the number of timesteps to generate
        :return: (np.ndarray, shape: (num_songs, numsteps, num_features)) an array of songs
        """
        # this needs to be fixed to use all the starter values
        rand = np.random.RandomState(int(time.time()))

        inputs = rand.normal(.5, .2, (num_songs, num_steps, 156))

        output = self.sess.run(self.G_sample, feed_dict={self.x: inputs, self.seq_len: [num_steps for i in range(num_songs)]})

        # set states to None in case generate Sequence is used

        return np.round(output).astype(int)

    def generate_midi_from_sequences(self, sequence, dir_path):
        """

        :param sequence: (np.ndarray, shape: (num_songs, numsteps, num_features)) an array of songs,
        like is outputed from self.generateSequence()
        :param dir_path: (string, path) the directory to save the songs to
        :return: None
        """
        for i in range(len(sequence)):
            mm.noteStateMatrixToMidi(sequence[i], dir_path + 'generated_chord_{}'.format(i))

    def trainAdversarially(self, training_expected, epochs, report_interval=10, seqlens=None, batch_size = 100):
        """

        :param training_input: 
        :param training_expected:
        :param epochs:
        :param report_interval:
        :param seqlens:
        :return:

        """

        tqdm.write('Beginning LSTM training for {0} epochs at report interval {1}'.format(epochs, report_interval))
        train_G = True
        train_D = True

        iter_ = tqdm(range(epochs), desc="{0}.learn".format(self.model_name), ascii=True)
        max_seqlen = max(map(len, training_expected))
        unbatched_training_expected = training_expected
        unbatched_seqlens = seqlens
        for i in iter_:


            rand = np.random.RandomState(int(time.time()))
            idx = np.arange(len(unbatched_training_expected))
            np.random.shuffle(idx)
            training_expected = [unbatched_training_expected[i] for i in idx]
            seqlens = [unbatched_seqlens[i] for i in idx]
            training_expected = split_list(training_expected, batch_size)
            seqlens = split_list(seqlens, batch_size)

            for k in tqdm(range(len(training_expected))):

                training_input = []
                for j in range(len(training_expected[k])):
                    training_input.append(rand.normal(.5, .2, (len(training_expected[k][j]), 156)))
                    if (len(training_expected[k][j]) < max_seqlen):
                        training_input[j] = np.pad(training_input[j],
                                               pad_width=(((0, max_seqlen - len(training_expected[k][j])), (0, 0))),
                                               mode='constant',
                                               constant_values=0)

                G_err = self.sess.run(self.G_loss, feed_dict={self.x: training_input, self.y: training_expected[k],
                                                          self.seq_len: seqlens[k]})
                D_err = self.sess.run(self.D_loss, feed_dict={self.x: training_input, self.y: training_expected[k],
                                                       self.seq_len: seqlens[k]})
                fake_count = self.sess.run(self.real_count, feed_dict={self.x: training_input, self.y: training_expected[k], self.seq_len: seqlens[k]})
                real_count = self.sess.run(self.fake_count, feed_dict={self.x: training_input, self.y: training_expected[k], self.seq_len: seqlens[k]})

                G_stop_count = 0
                D_stop_count = 0

                '''
                                if fake_count < .45:
                    print('stopping G')
                    train_G = False
                    G_stop_count += 1
                else:
                    train_G = True
                    G_stop_count = 0
                if fake_count > .55:
                    train_D = False
                    print('stopping D')
                    D_stop_count += 1
                else:
                    train_D = True
                    D_stop_count = 0

                if real_count > .9:
                    break
                '''


                if train_G:
                    self.sess.run('G_optimizer',
                          feed_dict={self.x: training_input, self.y: training_expected[k], self.seq_len: seqlens[k]})
                if train_D:
                    self.sess.run('D_optimizer',
                          feed_dict={self.x: training_input, self.y: training_expected[k], self.seq_len: seqlens[k]})


            if i % report_interval == 0:
                tqdm.write('Real Count {}'.format(real_count))

                tqdm.write('Fake Count {}'.format(fake_count))
                self._save((G_err, D_err), i, epochs)
                self._progress_sequence((G_err, D_err), i, epochs)
                tqdm.write('Sequence generated')
                tqdm.write('G Error {}'.format(
                     G_err))
                tqdm.write('D Error {}'.format(
                    D_err))



    def _save(self, err, i, epochs, save_dir='./model_saves'):
        try:
            g_err, d_err = err
            s_path = os.path.join(save_dir, self.model_name, 'G{0}_D{1}__{2}_{3}__{4}.ckpt'.format(g_err, d_err, i,
                        epochs, str(datetime.now()).replace(':', '_')))
            return self.saver.save(self.sess, s_path)
        except:
            s_path = os.path.join(save_dir, self.model_name, 'E{0}__{1}_{2}__{3}.ckpt'.format(err, i,
                                                                                                   epochs, str(
                    datetime.now()).replace(':', '_')))
            return self.saver.save(self.sess, s_path)

    def _progress_sequence(self, err, i, epochs, save_dir='./progress_sequences'):
        s_path = None
        try:
            g_err, d_err = err
            s_path = os.path.join(save_dir, self.model_name, 'G{0}_D{1}__{2}_{3}__{4}'.format(g_err, d_err, i,
                        epochs, str(datetime.now()).replace(':', '_')))
        except:
            s_path = os.path.join(save_dir, self.model_name, 'E{0}__{1}_{2}__{3}'.format(err, i,
                        epochs, str(datetime.now()).replace(':', '_')))
        os.makedirs(s_path, exist_ok=True)
        sequences = self.generate_sequence(1, 50)
        for i in range(len(sequences)):
            mm.noteStateMatrixToMidi(sequences[i], os.path.join(s_path, '{0}.mid'.format(i)))

