import tensorflow as tf
from tqdm import tqdm
import numpy as np
from glob import glob
import midi_manipulation

model_name = 'lstm_c02'
song_directory = './beeth'
learning_rate = .5
batch_size = 10
load_from_saved = True
epochs = 0
num_features = 156
layer_units = 156
n_steps = 100  # time steps
max_songs = 3
report_interval = 1

songs = midi_manipulation.get_songs(song_directory, model_name, max_songs)

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

states = None

def generator(inputs):
    with tf.variable_scope('generator_lstm_layer{0}'.format(1)):
        global states
        generator_outputs, states = tf.nn.dynamic_rnn(generator_lstm_cell, inputs, dtype=tf.float32,
                                                      sequence_length=seq_len, initial_state = states)

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

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake), name='D_loss')
G_loss = -tf.reduce_mean(tf.log(D_fake), name='G_loss')

D_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='D_optimizer').minimize(D_loss, var_list=D_vars)
G_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='G_optimizer').minimize(G_loss, var_list=G_vars)

saver = tf.train.Saver()

sess = tf.Session()
save_dir = './model_saves/{}/{}'.format(model_name, model_name)
if load_from_saved:
    saver.restore(sess, tf.train.latest_checkpoint('./model_saves/{}/'.format(model_name, model_name)))
else:
    init = tf.global_variables_initializer()
    sess.run(init)

def generateSequence(starter, numsteps):

    global states
    states = None

    output = [starter[0]]

    for i in tqdm(range(numsteps)):
        oneoutput = sess.run(G_sample, feed_dict={x: np.expand_dims(output[-1], 0), seq_len: [1]})[-1]
        print(states)
        output.append(oneoutput)

    return output

iter_ = tqdm(range(epochs), desc="{0}.learn".format(model_name))

for i in iter_:
    states = None
    sess.run('G_optimizer', feed_dict={x: input_sequence, y: expected_output, seq_len: seqlens})
    states = None
    sess.run('D_optimizer', feed_dict={x: input_sequence, y: expected_output, seq_len: seqlens})
    if i % report_interval == 0:
        #save
        print('G Error {}'.format(sess.run(G_loss, feed_dict={x: input_sequence, y: expected_output, seq_len: seqlens})))
        print('D Error {}'.format(
            sess.run(D_loss, feed_dict={x: input_sequence, y: expected_output, seq_len: seqlens})))

starter = np.transpose(input_sequence[:2][:10], (1, 0, 2))

print(generateSequence(starter, 100))

saver.save(sess, save_dir)

sess.close()

