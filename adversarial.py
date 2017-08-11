from datetime import datetime
from glob import iglob
import midi_manipulation as mm
import numpy as np
import os
from select import select
from sys import stdin
import tensorflow as tf
from tqdm import tqdm

# Comment this line to turn on TensorFlow warnings including possibilities for CPU optimization by use of
# newer instruction sets
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model architecture ------------------------------------------------------------------------------
#
# real data: X
# lstm_net_generator, multilayer
# lstm_net_discriminator, bidirectional, multilayer
#
# G_sample = lstm_net_generator.output(X)
# D_real = lstmnet2.output(X)
# D_fake = lstmnet2.output(G_sample)
#
# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))
#
# variables in generator: G_vars
# variables in discriminator: D_vars
#
# D_optimizer = tf.train.someoptimizer(D_lr).minimize(D_loss, var_list = G_vars)
# G_optimizer = tf.train.someoptimizer(G_lr).minimize(G_loss, var_list = D_vars)
#
# -------------------------------------------------------------------------------------------------


class AdversarialNet:
    """
        Implementation of a C-RNN-GAN on top of TensorFlow with learning, generation, saving/loading.
    """

    @classmethod
    def load_or_new(cls, model_name, *args, file_name=None, save_dir=os.path.join('', 'model_saves'), **kwargs):
        """
        "Normal" constructor - make a model if it doesn't exist or load a save if it does.
        :param model_name: the model name, used for naming folders in outputs
        :param args: arguments to pass to new() method
        :param file_name: if provided, load in a specific x.ckpt.meta file
        :param save_dir: where to look for the saved model
        :param kwargs: keyword arguments to pass to new() method
        :return: the new or loaded model
        """
        try:
            return cls.load(model_name, file_name, save_dir)
        except FileNotFoundError:
            return cls.new(model_name, *args, **kwargs)

    @classmethod
    def new(cls, model_name, num_features, layer_units, num_layers):
        """
        Build a new copy of the model.
        :param model_name: the model name, used for naming folders in outputs
        :param num_features: the number of features i.e. vocabulary size (156 for use with midi_manipulation)
        :param layer_units: the number of LSTM units per layer
        :param num_layers: the number of LSTM layers per cell
        :return: an untrained instance of AdversarialNet
        """
        with tf.name_scope('placeholders'):
            x = tf.placeholder(tf.float32, (None, None, num_features), name='x')
            seq_lens = tf.placeholder(tf.int32, (None,), name='seq_lens')
            g_learning_rate = tf.placeholder(tf.float32, name='g_learning_rate')
            d_learning_rate = tf.placeholder(tf.float32, name='d_learning_rate')

        with tf.variable_scope('generator'):
            w_out = tf.Variable(tf.truncated_normal([layer_units, num_features], stddev=.1), name='w')
            b_out = tf.Variable(tf.truncated_normal([num_features], stddev=.1), name='b')

            lstm_cell = cls._construct_cell(layer_units, num_layers)

            # associate the cell
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_lens)
            # run thru forward-fed layer
            generated_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, w_out) + b_out), outputs)
            # give it a name
            tf.identity(generated_outputs, name='output')

        with tf.variable_scope('discriminator'):
            w_out = tf.Variable(tf.truncated_normal([layer_units * 2, 1], stddev=.1), name='w')
            b_out = tf.Variable(tf.truncated_normal([1], stddev=.1), name='b')

            with tf.variable_scope('fw'):
                lstm_cell_fw = cls._construct_cell(layer_units, num_layers)
            with tf.variable_scope('bw'):
                lstm_cell_bw = cls._construct_cell(layer_units, num_layers)

            # FAKE -----------------------------------------------------------------------------------------------------
            # associate the cell with the fake inputs
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, generated_outputs,
                                                         dtype=tf.float32,
                                                         sequence_length=seq_lens)
            # concatenate the outputs
            outputs_fw, outputs_bw = outputs
            outputs = tf.concat([outputs_fw, outputs_bw], 2)
            # run thru forward-fed layer
            fake_outputs = tf.map_fn(lambda f_output: tf.sigmoid(tf.matmul(f_output, w_out) + b_out), outputs)
            # give it a name
            tf.identity(fake_outputs, name='fake_output')

            # REAL -----------------------------------------------------------------------------------------------------
            # associate the cell with the real inputs
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, x, dtype=tf.float32,
                                                         sequence_length=seq_lens)
            # concatenate the outputs
            outputs_fw, outputs_bw = outputs
            outputs = tf.concat([outputs_fw, outputs_bw], 2)
            # run thru forward-fed layer
            real_outputs = tf.map_fn(lambda r_output: tf.sigmoid(tf.matmul(r_output, w_out) + b_out), outputs)
            # give it a name
            tf.identity(real_outputs, name='real_output')

        with tf.name_scope('optimizers'):
            d_cost = tf.negative(tf.reduce_mean(tf.log(real_outputs) + tf.log(1. - fake_outputs)), name='d_cost')
            g_cost = tf.negative(tf.reduce_mean(tf.log(fake_outputs)), name='g_cost')

            # performance values: how the generator and discriminator are doing
            # we think real_perf should be close to 1 and fake_perf should be close to 0.5 but not sure
            real_perf = tf.reduce_mean(real_outputs, name='real_perf')
            fake_perf = tf.reduce_mean(fake_outputs, name='fake_perf')

            tf.train.RMSPropOptimizer(d_learning_rate, name='d_opt').minimize(d_cost,
                                                                              var_list=tf.get_collection(
                                                                                  tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                  scope='discriminator'))
            tf.train.RMSPropOptimizer(g_learning_rate, name='g_opt').minimize(g_cost,
                                                                              var_list=tf.get_collection(
                                                                                  tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                  scope='generator'))

        return cls(model_name)

    @classmethod
    def load(cls, model_name, file_name=None, save_dir=os.path.join('', 'model_saves')):
        """
        Load the model from a save file.
        :param model_name: the model name, used for naming folders in outputs
        :param file_name: optional: a particular .ckpt.meta file to load from
        :param save_dir: optional: where to look for the model saves (default './model_saves')
        :return: a trained instance of AdversarialNet
        """
        if file_name is None: # no name provided, use the most recent save
            try:
                # find the most recent save file
                selector = os.path.join(save_dir, model_name, '*.ckpt.meta')
                file_name = max(iglob(selector), key=os.path.getctime)
            except ValueError:
                raise FileNotFoundError("Model by name {0} does not exist...".format(model_name))
        else: # try to load a particular file
            file_name = os.path.join(save_dir, file_name)
            if not os.path.isfile(file_name):
                raise ValueError("Specific model file by name {0} does not exist...".format(model_name))
        return cls(model_name, restore_file=file_name)

    def __init__(self, model_name, restore_file=None):
        """
        *** PRIVATE CONSTRUCTOR. DO NOT INSTANTIATE DIRECTLY. USE CLASSMETHODS TO GET AN INSTANCE ***
        Initialize an instance of AdversarialNet. Assumes the graph given by tf.get_default_graph() contains a
        properly formatted model.
        :param model_name: the model name, used for naming folders in outputs
        :param restore_file: optional: a particular .ckpt.meta file to load from
        """
        self.managed = False
        self.model_name = model_name
        self.restore_file = restore_file
        self.trained = (self.restore_file is not None)

    def __enter__(self):
        """
        Begin use of this object as a context manager (with block). Initialize the tf.Session,
        tf.train.Saver, and tf.summary.FileWriter. Initialize the graph and variables appropriately.
        :return:
        """
        sess = tf.Session()
        if self.restore_file is not None:  # load from a save file .ckpt.meta
            # import_meta_graph wants a filename something.ckpt.meta
            saver = tf.train.import_meta_graph(self.restore_file)
            # restore wants a filename something.ckpt so cut off the .meta
            saver.restore(sess, self.restore_file[:-5])
            self.saver = saver
            self.trained = True
        else:
            self.saver = tf.train.Saver()
            # run the initializer/randomizer only if there's no save file; saver.restore will take care of it otherwise
            sess.run(tf.global_variables_initializer())
        self.sess = sess
        # make a tensorboard
        self.writer = tf.summary.FileWriter("tensorboard/{0}".format(self.model_name), self.sess.graph)
        self.managed = True
        return self

    def __exit__(self, type_, value, traceback):
        """
        End use of this object as a context manager (with block)
        :param type_: the type of the exception, if any
        :param value: the string value of the exception, if any
        :param traceback: the exception traceback, if any
        :return: None
        """
        # close everything to be safe and conserve resources
        self.managed = False
        self.saver = None
        self.writer.close()
        self.sess.close()

    def _save(self, g_err, d_err, i, epochs, save_dir='./model_saves'):
        """
        Save the model, using a filename descriptive of the current state.
        :param g_err: the generator error
        :param d_err: the discriminator error
        :param i: the iteration
        :param epochs: the total training session length
        :param save_dir: optional: where to save (default: './model_saves')
        :return: the final save path
        """
        assert self.managed

        s_path = os.path.join(save_dir, self.model_name,
            '{4}__G{0}_D{1}__{2}_{3}.ckpt'.format(g_err, d_err, i, epochs, str(datetime.now()).replace(':', '_')))
        # make sure the parent directories exist
        os.makedirs(s_path, exist_ok=True)
        # Saver.save returns the path
        return self.saver.save(self.sess, s_path)

    @staticmethod
    def _construct_cell(layer_units, num_layers):
        """
        Build a discretely scoped multilayer LSTM cell.
        :param layer_units: the number of LSTM units per layer
        :param num_layers: the number of LSTM layers in the cell
        :return: a tf.contrib.rnn.MultiRNNCell instance representing the cell
        """
        cell_list = []
        for i in range(num_layers):
            with tf.variable_scope('layer_{0}'.format(i)):
                cell_list.append(tf.contrib.rnn.LSTMCell(layer_units))
        return tf.contrib.rnn.MultiRNNCell(cell_list)

    def learn_multiple_epochs(self, x_seq, seq_lens, g_learning_rate, d_learning_rate, epochs, report_interval=None,
                              progress_bar=True):
        """
        Learn for multiple epochs.
        :param x_seq: the inputs
        :param seq_lens: a vector representing the length of each input used to trim off any padding present
        :param g_learning_rate: the generator learning rate
        :param d_learning_rate: the discriminator learning rate
        :param epochs: the total training session length
        :param report_interval: optional: the interval at which to report error information
         to stdout and generate samples (default: sqrt(epochs))
        :param progress_bar: optional: if True, display an ascii-based tqdm progress bar
        :return: None
        """
        report_interval = report_interval or epochs ** 0.5 # can't do in method signature because it references epochs

        assert self.managed

        # possibly use a progress bar
        iter_ = tqdm(range(epochs), desc="{0}.learn".format(self.model_name), ascii=True) if progress_bar else range(
            epochs)

        stops_cache = (0, 0)

        for i in iter_:
            errs, perfs, stops = self.learn_one_epoch(x_seq, seq_lens, g_learning_rate, d_learning_rate)
            # add the stops to the cache
            stops_cache = [sum(x) for x in zip(stops_cache, stops)]

            if i % report_interval == 0 or i + 1 == epochs: # also report if last epoch regardless of modulus
                self._save(errs[0], errs[1], i, epochs)
                self.generate(10, 100, i)
                tqdm.write('Generator Error: {0}'.format(errs[0]))
                tqdm.write('Discriminator Error: {0}'.format(errs[1]))
                tqdm.write('Real Performance: {0}'.format(perfs[0]))
                tqdm.write('Fake Performance: {0}'.format(perfs[1]))
                tqdm.write('Generator Stops: {0} (new: {1})'.format(stops_cache[0], str(bool(stops[0]))))
                tqdm.write('Discriminator Stops: {0} (new: {1})'.format(stops_cache[1], str(bool(stops[1]))))
                tqdm.write('generated 10/100')

    def learn_one_epoch(self, x_seq, seq_lens, g_learning_rate, d_learning_rate, train_g_threshold=0.53, train_d_threshold=0.45):
        """
        Learn for one epoch.
        :param x_seq: the inputs
        :param seq_lens: a vector representing the length of each input used to trim off any padding present
        :param g_learning_rate: the generator learning rate
        :param d_learning_rate: the discriminator learning rate
        :param train_g_threshold: the break threshold for fake peformance below which the generator will not train
        :param train_g_threshold: the break threshold for fake peformance above which the discriminator will not train
        :return: a Tuple ((generator_error, discriminator_err), (real_performance, fake_performance), (g_on, d_on))
        """
        assert self.managed

        feed_dict = {
            'placeholders/x:0': x_seq,
            'placeholders/seq_lens:0': seq_lens,
            'placeholders/g_learning_rate:0': g_learning_rate,
            'placeholders/d_learning_rate:0': d_learning_rate
        }

        g_error, d_error = self.sess.run(['optimizers/g_cost:0', 'optimizers/d_cost:0'], feed_dict=feed_dict)

        real_perf, fake_perf = self.sess.run(['optimizers/real_perf:0', 'optimizers/fake_perf:0'], feed_dict=feed_dict)

        train_G = not fake_perf > train_g_threshold
        train_D = not fake_perf < train_d_threshold
        if real_perf > .9:
            raise ValueError('real_perf > 0.9')

        if train_G:
            self.sess.run('optimizers/g_opt', feed_dict=feed_dict)
        if train_D:
            self.sess.run('optimizers/d_opt', feed_dict=feed_dict)

        self.trained = True
        return (g_error, d_error), (real_perf, fake_perf), (int(not train_G), int(not train_D))

    def learn_interactive(self, x_seq, seq_lens):
        """
        Learn continuously with input accepted from stdin to tune parameters and produce samples on-demand.
        Prompts for initial learning rates and report interval.
        :param x_seq: the inputs
        :param seq_lens: a vector representing the length of each input used to trim off any padding present
        :return: None
        """
        assert self.managed

        epoch = 0
        print('-- Beginning interactive learning --')
        g_learning_rate = float(input("Initial g_learning_rate: "))
        d_learning_rate = float(input("Initial d_learning_rate: "))
        report_interval = int(input("Initial report interval: "))
        train_g_threshold = 0.53
        train_d_threshold = 0.45
        stops_cache = (0, 0)

        while True: # broken by return statement
            print("Epoch: {0}".format(epoch))
            if stdin in select([stdin], [], [], 0)[0]: # if something has been typed and \n-ed on stdin
                line = stdin.readline()
                if line:  # read something that's not EOF
                    if line[:2] == "gl":
                        g_learning_rate = float(input("New g_learning_rate: "))
                    elif line[:2] == "dl":
                        d_learning_rate = float(input("New d_learning_rate: "))
                    elif line[:2] == "gt":
                        train_g_threshold = float(input("New train_g threshold: "))
                    elif line[:2] == "dt":
                        train_d_threshold = float(input("New train_d threshold: "))
                    elif line[0] == "r":
                        report_interval = int(input("New report interval: "))
                    elif line[0] == "s":
                        self.generate(10, 100, epoch)
                        print("generated 10/100")
                else:  # an empty line means stdin has been closed/EOF/Ctrl-D
                    # do stuff to shut down and final report
                    self._save(errs[0], errs[1], epoch, 0)
                    self.generate(10, 100, epoch)

                    print('Generator Error: {0}'.format(errs[0]))
                    print('Discriminator Error: {0}'.format(errs[1]))
                    print('Real Performance: {0}'.format(perfs[0]))
                    print('Fake Performance: {0}'.format(perfs[1]))
                    print('Generator Stops: {0} (new: {1})'.format(stops_cache[0], str(bool(stops[0]))))
                    print('Discriminator Stops: {0} (new: {1})'.format(stops_cache[1], str(bool(stops[1]))))
                    print('generated 10/100')

                    print("-- Ending interactive learning --")
                    return  # breaks
            else:  # nothing available on stdin, keep going
                errs, perfs, stops = self.learn_one_epoch(x_seq, seq_lens, g_learning_rate, d_learning_rate,
                                                      train_g_threshold, train_d_threshold)
                # add the stops to the cache
                stops_cache = [sum(x) for x in zip(stops_cache, stops)]
                if epoch % report_interval == 0:
                    self._save(errs[0], errs[1], epoch, 0)
                    self.generate(10, 100, epoch)

                    print('Generator Error: {0}'.format(errs[0]))
                    print('Discriminator Error: {0}'.format(errs[1]))
                    print('Real Performance: {0}'.format(perfs[0]))
                    print('Fake Performance: {0}'.format(perfs[1]))
                    print('Generator Stops: {0} (new: {1})'.format(stops_cache[0], str(bool(stops[0]))))
                    print('Discriminator Stops: {0} (new: {1})'.format(stops_cache[1], str(bool(stops[1]))))
                    print('generated 10/100')

            epoch += 1

    def generate(self, num_samples, timestamps_per_sample, epoch=None, save_dir='./progress_sequences'):
        """
        Generate output sequences from a trained model.
        :param num_samples: the number of midi files
        :param timestamps_per_sample: the number of timestamps to generate for each sample
        :param epoch: optional: an epoch number to label the folder with
        :param save_dir: optional: where to save (default: './progress_sequences')
        :return: None
        """
        assert self.managed
        assert self.trained

        # Get the number of features from the x placeholder
        num_features = tf.get_default_graph().get_tensor_by_name('placeholders/x:0').get_shape().as_list()[2]

        # Make some random noise to go in the x placeholder
        noise = np.random.normal(.5, .2, (num_samples, timestamps_per_sample, num_features))

        feed_dict = {
            'placeholders/x:0': noise,
            'placeholders/seq_lens:0': [timestamps_per_sample for _ in range(num_samples)]
        }

        output = self.sess.run('generator/output:0', feed_dict=feed_dict)

        # Convert probabilities to 0s and 1s for noteStateMatrixToMidi
        output = np.round(output).astype(int)

        s_path = os.path.join(save_dir, self.model_name,
                              'Epoch{0}__{1}'.format(epoch, str(datetime.now()).replace(':', '_')) if epoch else
                              str(datetime.now()).replace(':', '_'))

        # make sure the parent directories exist
        os.makedirs(s_path, exist_ok=True)

        for i in range(len(output)):
            mm.noteStateMatrixToMidi(output[i], os.path.join(s_path, str(i)))
