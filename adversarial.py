from datetime import datetime
from glob import iglob
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import midi_manipulation as mm

# real data: X
# lstmnet1
# lstmnet2, bidirectional?

# G_sample = lstmnet1.output(X)
# D_real = lstmnet2.output(X)
# D_fake = lstmnet2.output(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# variables in generator: G_vars
# variables in discriminator: D_vars

# D_optimizer = tf.train.someoptimizer(D_lr).minimize(D_loss, var_list = G_vars)
# G_optimizer = tf.train.someoptimizer(G_lr).minimize(G_loss, var_list = D_vars)

class C_RNN_GAN_Factory:
    @classmethod
    def load_or_new(cls, model_name, learning_rate, num_features, layer_units, num_steps, file_name=None,
                    save_dir=os.path.join('', 'model_saves')):
        try:
            return cls.load(model_name, file_name, save_dir)
        except FileNotFoundError:
            return cls.new(model_name, learning_rate, num_features, layer_units, num_steps)

    @classmethod
    def new(cls, model_name, learning_rate, num_features, layer_units, num_steps):

        x = tf.placeholder(tf.float32, (None, num_steps, num_features), name='x')
        y = tf.placeholder(tf.float32, (None, num_steps, num_features), name='y')

        w_out = tf.Variable(tf.truncated_normal([layer_units, num_features], stddev=.1), name='w')
        b_out = tf.Variable(tf.truncated_normal([num_features], stddev=.1), name='b')

        seq_len = tf.placeholder(tf.int32, (None,), name='seq_lens')
        with tf.variable_scope('generator_lstm_layer{0}'.format(1)):
            generator_lstm_cell = tf.contrib.rnn.LSTMCell(layer_units)
        def generator(inputs):
            with tf.variable_scope('generator_lstm_layer{0}'.format(1)):
                generator_outputs, states = tf.nn.dynamic_rnn(generator_lstm_cell, inputs, dtype=tf.float32, sequence_length=seq_len)
            generator_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, w_out) + b_out), generator_outputs, name='y_')
            return generator_outputs

        with tf.variable_scope('discriminator_lstm_layer{0}'.format(1)):
            discriminator_lstm_cell = tf.contrib.rnn.LSTMCell(layer_units)
        def discriminator(inputs):
            with tf.variable_scope('discriminator_lstm_layer{0}'.format(1)):
                discriminator_outputs, states = tf.nn.dynamic_rnn(discriminator_lstm_cell, inputs, dtype=tf.float32, sequence_length=seq_len)
            discriminator_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, w_out) + b_out), discriminator_outputs, name='y_')
            return discriminator_outputs

        G_sample = generator(x)
        D_real = discriminator(x)
        D_fake = discriminator(G_sample)

        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        G_loss = -tf.reduce_mean(tf.log(D_fake))

        D_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='D_optimizer').minimize(D_loss)
        G_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='G_optimizer').minimize(G_loss)

        return cls._C_RNN_GAN(model_name, generator_cell=generator_lstm_cell, discriminator_cell=discriminator_lstm_cell, generator=G_sample)

    @classmethod
    def load(cls, model_name, file_name=None, save_dir=os.path.join('', 'model_saves')):
        """
        Load a previously saved neural net for a particular model. Behavior is undefined if the model was not
        output by an instance of NeuralNet.
        :param model_name: The model name.
        :param file_name: Optional: The path *RELATIVE TO save_dir* to the .ckpt.meta file for the particular save file
        collection. If unspecified, the most recent save for the model is used.
        :param save_dir: The relative or absolute path to the directory where the model directory is located.
        If unspecified, defaults to './model_saves'.'.
        :return: A trained instance of NeuralNet.
        """
        if file_name is None:
            try:
                selector = os.path.join(save_dir, model_name, '*.ckpt.meta')
                file_name = max(iglob(selector), key=os.path.getctime)
            except ValueError:
                raise FileNotFoundError("This model does not exist...")
        else:
            file_name = os.path.join(save_dir, file_name)
            if not os.path.isfile(file_name):
                raise ValueError("Model file by name {0} does not exist...".format(model_name))
        return cls._C_RNN_GAN(model_name, restore_file=file_name)

class C_RNN_GAN:
    def __init__(self, model_name, restore_file=None, generator_cell=None, discriminator_cell=None, generator=None):
        """
        ***PRIVATE CONSTRUCTOR. DO NOT INSTANTIATE DIRECTLY. USE NeuralNet.new or NeuralNet.load.***
        Stores model information and possible file location in preparation for creating session with __enter__.
        :param model_name: the model name
        :param restore_file: a restore file path if the model should be restored; defaults to None.
        """
        self.managed = False
        self.model_name = model_name
        self.restore_file = restore_file
        self._generator_cell = generator_cell
        self._discriminator = discriminator_cell
        self._generator = generator
        self.trained = (self.restore_file is not None)

    def __enter__(self):
        """
        Enter a managed-resource block (with block). Start (and populate, if necessary) a tf.Session and tf.Saver.
        :return: self
        """
        sess = tf.Session()
        if self.restore_file is not None:
            saver = tf.train.import_meta_graph(self.restore_file)
            saver.restore(sess, self.restore_file[:-5])
            layer_size = [var for var in tf.global_variables() if var.name == 'lstm_layer1/rnn/lstm_cell/bias:0'][0].get_shape().as_list()[0] // 4
            self._generator_cell = tf.contrib.rnn.LSTMCell(layer_size)
            self._discriminator_cell = tf.contrib.rnn.LSTMCell(layer_size)
            self.saver = saver
            self.trained = True
        else:
            self.saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
        self.sess = sess
        self.managed = True
        return self

    def __exit__(self, type_, value, traceback):
        """
        Exit a managed-resource block (with block). End the tf.Session and tf.Saver.
        :param type_: exception type
        :param value: exception value
        :param traceback: exception traceback
        """
        self.managed = False
        self.saver = None
        self.sess.close()

    def _save(self, err, i, epochs, save_dir='./model_saves'):
        """
        Write out the current state to a checkpoint file in save_dir. The name is formatted as 'err__i_epochs__datetime'
        where err is the current error as measured by error_metric, i is the current epoch, and epochs is the total
        number of epochs.
        :param save_dir: The location to save the files.
        :param err: The current error, as measured by error_metric.
        :param i: The current epoch.
        :param epochs: The total number of epochs to be executed in this learn() call.
        :return: The path to the saved file.
        :raises RuntimeError: if the TFRunner instance is not resource managed.
        """
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        s_path = os.path.join(save_dir, self.model_name, '{0}__{1}_{2}__{3}.ckpt'.format(err, i, epochs,
                                                                                         str(
                                                                                             datetime.now()).replace(
                                                                                             ':',
                                                                                             '_')))
        return self.saver.save(self.sess, s_path)

    def learn(self, xseq, yseq, seqlens, epochs, report_interval=1000, progress_bar=True):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            iter_ = tqdm(range(epochs), desc="{0}.learn".format(self.model_name)) if progress_bar else range(epochs)
            for i in iter_:
                self.sess.run('optimizer', feed_dict={'x:0': xseq, 'y:0': yseq, 'seq_lens:0': seqlens})
                if i % report_interval == 0:
                    measured_error = self._error(xseq, yseq, seqlens)
                    self._save(measured_error, i, epochs)
                    print(measured_error)
        self.trained = True

    def _error(self, xseq, yseq, seqlens):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            return self.sess.run('cost:0', feed_dict={'x:0': xseq, 'y:0': yseq, 'seq_lens:0': seqlens})

    def validate(self, xseq, yseq, seqlens):
        # TODO: Meaningful???
        if not self.trained:
            raise RuntimeError("attempted to call validate() on untrained model")
        else:
            return self._error(xseq, yseq, seqlens)

    def feed_forward(self, inputs, state):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            if not self.trained:
                raise RuntimeError("attempted to call feed_foward() on untrained model")
            else:
                return self._generator_cell(inputs, state)

    def generate_music_sequences_from_noise(self, num_timesteps, num_songs):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            if not self.trained:
                raise RuntimeError("attempted to call _generate_music() on untrained model")
            else:
                xseq = tf.truncated_normal((num_songs, num_timesteps, 156), mean=.5, stddev=.1)
                seqlens = np.empty([num_songs])
                seqlens.fill(num_timesteps)
                return self.sess.run('y_', feed_dict={'x:0': xseq.eval(session=self.sess), 'seq_lens:0': seqlens})

    '''def generate_music_sequences_recursively(self, num_timesteps, num_songs, starter, starter_length, layer_units):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            if not self.trained:
                raise RuntimeError("attempted to call generate_music_sequences_recursively() on untrained model")
            else:
                #input_ = tf.cast(starter, dtype=np.float32)[-1]
                #input_ = starter = t
                with tf.variable_scope('lstm_layer1/rnn', reuse=True):
                    outputs = [tf.zeros([1, 156]),]
                    state = self._cell.zero_state(1, tf.float32)
                    for _ in tqdm(range(num_timesteps)):
                        o, s = self._cell(outputs[-1], state)
                        state = s
                        outputs.append(o)
                    return self.sess.run(outputs[-1])'''

    def generate_music_sequences_recursively(self, num_timesteps, num_songs, starter, starter_length, layer_units):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            if not self.trained:
                raise RuntimeError("attempted to call generate_music_sequences_recursively() on untrained model")
            else:

                outputs = starter
                for i in tqdm(range(num_timesteps)):
                    next = np.squeeze(self.sess.run(self._generator, feed_dict={'x:0': outputs[i]}))
                    print(next)
                    np.append(outputs, next)

                return np.transpose(outputs, (1, 0, 2))

    def generate_midi_from_sequences(self, sequence, dir_path):
        for i in range(len(sequence)):
            mm.noteStateMatrixToMidi(sequence[i], dir_path+'generated_chord_{}'.format(i))

C_RNN_GAN_Factory._C_RNN_GAN = C_RNN_GAN
del C_RNN_GAN

