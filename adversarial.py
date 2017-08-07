from datetime import datetime
from glob import iglob
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import midi_manipulation as mm

verbose = True

if not verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#real data: X
#lstmnet1
#lstmnet2, bidirectional?

# G_sample = lstmnet1.output(X)
# D_real = lstmnet2.output(X)
# D_fake = lstmnet2.output(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# variables in generator: G_vars
# variables in discriminator: D_vars

# D_optimizer = tf.train.someoptimizer(D_lr).minimize(D_loss, var_list = G_vars)
# G_optimizer = tf.train.someoptimizer(G_lr).minimize(G_loss, var_list = D_vars)

def custom_getter(default_getter, name, *args, **kwargs):
    try:
        return default_getter(name, *args, **kwargs)
    except ValueError:
        try:
            return [var for var in tf.global_variables() if var.name == name + ':0']
        except ValueError:
            raise ValueError('Variable {0} not found. All variables: '.format(name, [var.name for var in tf.global_variables()]))


class AdversarialNet:
    @classmethod
    def load_or_new(cls, model_name, learning_rate, num_features, layer_units, num_steps, file_name=None,
                    save_dir=os.path.join('', 'model_saves')):
        try:
            return cls.load(model_name, file_name, save_dir)
        except FileNotFoundError:
            return cls.new(model_name, learning_rate, num_features, layer_units, num_steps)

    @classmethod
    def new(cls, model_name, learning_rate, num_features, layer_units, num_steps):
        x = tf.placeholder(tf.float32, (None, None, num_features), name='x')
        y = tf.placeholder(tf.float32, (None, None, num_features), name='y')
        seq_lens = tf.placeholder(tf.int32, (None,), name='seq_lens')

        with tf.variable_scope('generator') as scope:
            w_out = tf.Variable(tf.truncated_normal([layer_units, num_features], stddev=.1), name='w')
            b_out = tf.Variable(tf.truncated_normal([num_features], stddev=.1), name='b')

            lstm_cell = tf.contrib.rnn.LSTMCell(layer_units)

            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_lens)
            gen_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, w_out) + b_out), outputs, name='g_')

        with tf.variable_scope('discriminator') as scope:
            w_out = tf.Variable(tf.truncated_normal([layer_units, num_features], stddev=.1), name='w')
            b_out = tf.Variable(tf.truncated_normal([num_features], stddev=.1), name='b')

            lstm_cell = tf.contrib.rnn.LSTMCell(layer_units)

            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_lens)
            real_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, w_out) + b_out), outputs, name='r_')

            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, gen_outputs, dtype=tf.float32, sequence_length=seq_lens)
            fake_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, w_out) + b_out), outputs, name='f_')

        with tf.variable_scope('optimizers') as scope:
            d_cost = tf.identity(-tf.reduce_mean(tf.log(real_outputs) + tf.log(1. - fake_outputs)), name='d_cost')
            g_cost = tf.identity(-tf.reduce_mean(tf.log(fake_outputs)), name='g_cost')

            d_optimizer = tf.train.RMSPropOptimizer(learning_rate, name='d_opt').minimize(d_cost) # varlist
            g_optimizer = tf.train.RMSPropOptimizer(learning_rate, name='g_opt').minimize(g_cost) # varlist

        return cls(model_name)

    @classmethod
    def load(cls, model_name, file_name=None, save_dir=os.path.join('', 'model_saves')):
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
        return cls(model_name, restore_file=file_name)

    def __init__(self, model_name, restore_file=None):
        self.managed = False
        self.model_name = model_name
        self.restore_file = restore_file
        self.trained = (self.restore_file is not None)

    def __enter__(self):
        sess = tf.Session()
        if self.restore_file is not None:
            saver = tf.train.import_meta_graph(self.restore_file)
            saver.restore(sess, self.restore_file[:-5])
            self.saver = saver
            self.trained = True
        else:
            self.saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
        self.sess = sess
        self.managed = True
        return self

    def __exit__(self, type_, value, traceback):
        self.managed = False
        self.saver = None
        self.sess.close()

    def _save(self, g_err, d_err, i, epochs, save_dir='./model_saves'):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        s_path = os.path.join(save_dir, self.model_name, 'G{0}_D{1}__{2}_{3}__{4}.ckpt'.format(g_err, d_err, i, epochs,
                                                                                         str(
                                                                                             datetime.now()).replace(
                                                                                             ':',
                                                                                             '_')))
        return self.saver.save(self.sess, s_path)

    def learn(self, xseq, yseq, seqlens, epochs, report_interval=None, progress_bar=True):
        report_interval = report_interval or epochs ** 0.5
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            iter_ = tqdm(range(epochs), desc="{0}.learn".format(self.model_name)) if progress_bar else range(epochs)
            for i in iter_:
                self.sess.run('optimizers/g_opt', feed_dict={'x:0': xseq, 'y:0': yseq, 'seq_lens:0': seqlens})
                self.sess.run('optimizers/d_opt', feed_dict={'x:0': xseq, 'y:0': yseq, 'seq_lens:0': seqlens})
                if i % report_interval == 0:
                    g_error = self.sess.run('optimizers/g_cost', feed_dict={'x:0': xseq, 'y:0': yseq, 'seq_lens:0': seqlens})
                    d_error = self.sess.run('optimizers/d_cost', feed_dict={'x:0': xseq, 'y:0': yseq, 'seq_lens:0': seqlens})
                    self._save(g_error, d_error, i, epochs)
                    print('Generator Error: {0}'.format(g_error))
                    print('Discriminator Error: {0}'.format(d_error))
        self.trained = True

    def generate(self, starter, iterations, progress_bar=True):
        output = [starter[0]]
        iter_ = tqdm(range(iterations), desc="{0}.generate".format(self.model_name) if progress_bar else range(iterations))
        for i in iter_:
            next_ = self.sess.run('generator/g_', feed_dict={'x:0': np.expand_dims(output[-1], 0), 'seq_lens:0': [1]})[-1]
            print(next_)
            output.append(next_)
        return output


