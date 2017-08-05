import tensorflow as tf
from tqdm import tqdm
import numpy as np
from glob import iglob
from glob import glob
from midi_manipulation import midiToNoteStateMatrix
import os

verbose = True

if not verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_name = 'lstm_b013'
song_directory = './beeth'
learning_rate = .5
batch_size = 10
epochs = 0
num_features = 156
layer_units = 156
n_steps = 100  # time steps
max_songs = 3
report_interval = 1

def get_songs(path, max=None):
    '''
    :param path: path to the songs directory
    :return: array of songs w/ timestamp events
    '''
    files = glob('{}/*.mid*'.format(path))
    files = files[:max] if max is not None else files
    songs = []
    c = 0
    for f in tqdm(files, desc='{0}.get_songs({1})'.format(model_name, path)):
        try:

            song = np.array(midiToNoteStateMatrix(f))
            songs.append(song)

        except Exception as e:
            raise e
    return songs

songs = get_songs(song_directory, max_songs)

input_sequence = []
expected_output = []
seqlens = []
max_seqlen = max(map(len, songs))

for song in tqdm(songs, desc="{0}.pad/seq".format(model_name)):
    seqlens.append(len(song) - 1)
    if (len(song) < max_seqlen):
        song = np.pad(song, pad_width=(((0, max_seqlen - len(song)), (0, 0))), mode='constant', constant_values=0)

    input_sequence.append(song[0:len(song) - 2])
    expected_output.append(song[1:len(song) - 1])


file_name = None
save_dir = None
'''
try:
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

    restore_file=file_name
    sess = tf.Session()
    if restore_file is not None:
        saver = tf.train.import_meta_graph(restore_file)
        saver.restore(sess, restore_file[:-5])
        layer_size = [var for var in tf.global_variables() if var.name == 'lstm_layer1/rnn/lstm_cell/bias:0'][
                         0].get_shape().as_list()[0] // 4
        cell = tf.contrib.rnn.LSTMCell(layer_size)
        saver = saver
        trained = True
    else:
        self.saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
    self.sess = sess
    self.managed = True
    return self
    
except FileNotFoundError:
    return cls.new(model_name, learning_rate, num_features, layer_units, num_steps)


'''

x = tf.placeholder(tf.float32, (None, None, num_features), name='x')
y = tf.placeholder(tf.float32, (None, None, num_features), name='y')


G_W1 = tf.Variable(tf.truncated_normal([layer_units, num_features], stddev=.1), name='G_W1')
G_b1 = tf.Variable(tf.truncated_normal([num_features], stddev=.1), name='G_b1')

G_vars = [G_W1, G_b1]

D_W1 = tf.Variable(tf.truncated_normal([layer_units, 1], stddev=.1), name='D_W1')
D_b1 = tf.Variable(tf.truncated_normal([1], stddev=.1), name='D_b1')

D_vars = [D_W1, D_b1]

seq_len = tf.placeholder(tf.int32, (None,), name='seq_lens')

with tf.variable_scope('generator_lstm_layer{0}'.format(1)):
    generator_lstm_cell = tf.contrib.rnn.LSTMCell(layer_units)

def generator(inputs):
    with tf.variable_scope('generator_lstm_layer{0}'.format(1)):
        generator_outputs, states = tf.nn.dynamic_rnn(generator_lstm_cell, inputs, dtype=tf.float32,
                                                      sequence_length=seq_len)
    generator_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, G_W1) + G_b1), generator_outputs,
                                  name='G_')
    return generator_outputs

with tf.variable_scope('discriminator_lstm_layer{0}'.format(1)):
    discriminator_lstm_cell = tf.contrib.rnn.LSTMCell(layer_units)

def discriminator(inputs):
    with tf.variable_scope('discriminator_lstm_layer{0}'.format(1)):
        discriminator_outputs, states = tf.nn.dynamic_rnn(discriminator_lstm_cell, inputs, dtype=tf.float32,
                                                          sequence_length=seq_len)
    discriminator_outputs = tf.map_fn(lambda output: tf.sigmoid(tf.matmul(output, D_W1) + D_b1),
                                      discriminator_outputs, name='D_')
    return discriminator_outputs

G_sample = generator(x)
D_real = discriminator(x)
D_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

D_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='D_optimizer').minimize(D_loss, var_list=D_vars)
G_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='G_optimizer').minimize(G_loss, var_list=G_vars)

sess = tf.Session()

def generateSequence(starter, numsteps):

    output = [starter[0]]

    for i in tqdm(range(numsteps)):
        oneoutput = sess.run(G_sample, feed_dict={x: np.expand_dims(output[-1], 0), seq_len: [1]})[-1]
        print(oneoutput)
        output.append(oneoutput)

    return output

iter_ = tqdm(range(epochs), desc="{0}.learn".format(model_name))

init = tf.global_variables_initializer()
sess.run(init)

for i in iter_:
    sess.run('G_optimizer', feed_dict={x: input_sequence, y: expected_output, seq_len: seqlens})
    sess.run('D_optimizer', feed_dict={x: input_sequence, y: expected_output, seq_len: seqlens})
    if i % report_interval == 0:
        #save
        print('G Error {}'.format(sess.run('G_loss', feed_dict={x: input_sequence, y: expected_output, seq_len: seqlens})))
        print('D Error {}'.format(
            sess.run('D_loss', feed_dict={x: input_sequence, y: expected_output, seq_len: seqlens})))

starter = np.transpose(input_sequence[:2][:10], (1, 0, 2))

print(generateSequence(starter, 100))

sess.close()

