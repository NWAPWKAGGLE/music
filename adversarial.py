from datetime import datetime
from glob import iglob
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import midi_manipulation as mm
from sys import stdin
from select import select

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
            raise ValueError('Variable {0} not found. All variables: '.format(name,
                                                                              [var.name for var in tf.global_variables()]))


class AdversarialNet:
    @classmethod
    def load_or_new(cls, model_name, learning_rate, num_features, layer_units, num_layers, file_name=None,
                    save_dir=os.path.join('', 'model_saves')):
        try:
            return cls.load(model_name, file_name, save_dir)
        except FileNotFoundError:
            return cls.new(model_name, learning_rate, num_features, layer_units, num_layers)

    @classmethod
    def new(cls, model_name, num_features, layer_units, num_layers):
        with tf.name_scope('placeholders'):
            x = tf.placeholder(tf.float32, (None, None, num_features), name='x')
            y = tf.placeholder(tf.float32, (None, None, num_features), name='y')
            seq_lens = tf.placeholder(tf.int32, (None,), name='seq_lens')
            g_learning_rate = tf.placeholder(tf.float32, name='g_learning_rate')
            d_learning_rate = tf.placeholder(tf.float32, name='d_learning_rate')


        with tf.variable_scope('generator') as scope:
            w_out = tf.Variable(tf.truncated_normal([layer_units, num_features], stddev=.1), name='w')
            b_out = tf.Variable(tf.truncated_normal([num_features], stddev=.1), name='b')

            lstm_cell = cls._construct_cell(layer_units, num_layers)

            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_lens)
            generated_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, w_out) + b_out), outputs, name='output')

        with tf.variable_scope('discriminator') as scope:
            w_out = tf.Variable(tf.truncated_normal([layer_units * 2, 1], stddev=.1), name='w')
            b_out = tf.Variable(tf.truncated_normal([1], stddev=.1), name='b')

            with tf.variable_scope('fw'):
                lstm_cell_fw = cls._construct_cell(layer_units, num_layers)
            with tf.variable_scope('bw'):
                lstm_cell_bw = cls._construct_cell(layer_units, num_layers)

            with tf.variable_scope('real'):
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, x, dtype=tf.float32,
                                                             sequence_length=seq_lens)
                outputs_fw, outputs_bw = outputs
                outputs = tf.concat([outputs_fw, outputs_bw], 2)
                real_outputs = tf.map_fn(lambda r_output: tf.sigmoid(tf.matmul(r_output, w_out) + b_out), outputs,
                                         name='output')

            with tf.variable_scope('fake'):
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, generated_outputs, dtype=tf.float32,
                                                             sequence_length=seq_lens)
                outputs_fw, outputs_bw = outputs
                outputs = tf.concat([outputs_fw, outputs_bw], 2)
                fake_outputs = tf.map_fn(lambda f_output: tf.sigmoid(tf.matmul(f_output, w_out) + b_out), outputs,
                                         name='output')

        with tf.name_scope('optimizers') as scope:
            d_cost = tf.identity(tf.reduce_mean(tf.log(real_outputs) + tf.log(1. - fake_outputs)), name='d_cost')
            g_cost = tf.identity(tf.reduce_mean(tf.log(fake_outputs)), name='g_cost')

            d_optimizer = tf.train.RMSPropOptimizer(d_learning_rate, name='d_opt').minimize(d_cost,
                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
            g_optimizer = tf.train.RMSPropOptimizer(g_learning_rate, name='g_opt').minimize(g_cost,
                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

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
        self.writer = tf.summary.FileWriter("tensorboard", self.sess.graph)
        self.managed = True
        return self

    def __exit__(self, type_, value, traceback):
        self.managed = False
        self.saver = None
        self.writer.close()
        self.sess.close()

    def _save(self, g_err, d_err, i, epochs, save_dir='./model_saves'):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:

            s_path = os.path.join(save_dir, self.model_name, 'G{0}_D{1}__{2}_{3}__{4}.ckpt'.format(g_err, d_err, i,
                        epochs, str(datetime.now()).replace(':', '_')))
            os.makedirs(s_path, exist_ok=True)
            return self.saver.save(self.sess, s_path)

    @staticmethod
    def _construct_cell(layer_units, num_layers):
        cell_list = []
        for i in range(num_layers):
            with tf.variable_scope('layer_{0}'.format(i)) as scope:
                cell_list.append(tf.contrib.rnn.LSTMCell(layer_units))
        return tf.contrib.rnn.MultiRNNCell(cell_list)

    def learn_multiple_epochs(self, x_seq, seq_lens, g_learning_rate, d_learning_rate, epochs, report_interval=None, progress_bar=True):
        report_interval = report_interval or epochs ** 0.5

        assert self.managed

        iter_ = tqdm(range(epochs), desc="{0}.learn".format(self.model_name), ascii=True) if progress_bar else range(epochs)
        for i in iter_:
            g_error, d_error = self.learn_one_epoch(x_seq, seq_lens, g_learning_rate, d_learning_rate)
            if i % report_interval == 0 or i + 1 == epochs:
                self._save(g_error, d_error, i, epochs)
                self.generate(10, 100, i)
                print('Generator Error: {0}\nDiscriminator Error: {1}\ngenerated 10/100'.format(g_error, d_error))

    def learn_one_epoch(self, x_seq, seq_lens, g_learning_rate, d_learning_rate):
        feed_dict = {
            'placeholders/x:0': x_seq,
            'placeholders/seq_lens:0': seq_lens,
            'placeholders/g_learning_rate:0': g_learning_rate,
            'placeholders/d_learning_rate:0': d_learning_rate
        }

        g_error, d_error = self.sess.run(['optimizers/g_cost', 'optimizers/d_cost'], feed_dict=feed_dict)

        if not g_error < .7 * d_error:
            self.sess.run('optimizers/g_opt', feed_dict=feed_dict)
        if not d_error < .7 * g_error:
            self.sess.run('optimizers/d_opt', feed_dict=feed_dict)

        self.trained = True

        return g_error, d_error

    def learn_interactive(self, x_seq, seq_lens):
        assert self.managed
        exiting = False
        epoch = 0
        g_learning_rate = float(input('Initial g_learning_rate: '))
        d_learning_rate = float(input('Initial d_learning_rate: '))
        while stdin in select([stdin], [], [], 0)[0]:
            epoch += 1
            print('Epoch: {0}'.format(epoch))
            line = stdin.readline()
            if line: # something that's not EOF
                if line[0] == 'g':
                    g_learning_rate = float(input('New g_learning_rate: '))
                elif line[0] == 'd':
                    d_learning_rate = float(input('New d_learning_rate: '))
                elif line[0] == 's':
                    self.generate(10, 100, epoch)
                    print('generated 10/100')
            else:  # an empty line means stdin has been closed
                self._save(0, 0, epoch, 0)
                return
        else:
            epoch += 1
            self.learn_one_epoch(x_seq, seq_lens, g_learning_rate, d_learning_rate)


    def generate(self, num_samples, timestamps_per_sample, epoch, save_dir='./progress_sequences'):
        assert self.managed
        assert self.trained

        # Get the number of features from the x placeholder
        num_features = tf.get_default_graph().get_tensor_by_name('x:0').get_shape().as_list()[2]

        # Make some random noise to go in the x placeholder
        noise = np.random.normal(.5, .2, (num_samples, timestamps_per_sample, num_features))

        feed_dict = {
            'placeholders/x:0': noise,
            'placeholders/seq_lens:0': np.array(num_samples).fill(timestamps_per_sample)
        }

        output = self.sess.run('generator/output', feed_dict=feed_dict)

        output = np.round(output).astype(int)

        s_path = os.path.join(save_dir, self.model_name, 'Epoch{0}__{1}'.format(epoch, str(datetime.now()).replace(
                                                                                            ':', '_')))
        os.makedirs(s_path, exist_ok=True)
        for i in range(len(output)):
            mm.noteStateMatrixToMidi(output[i], os.path.join(s_path, str(i)))


