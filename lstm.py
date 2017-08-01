import tensorflow as tf
import glob
from tqdm import tqdm
import numpy as np
from midi_manipulation import midiToNoteStateMatrix
from midi_manipulation import noteStateMatrixToMidi

np.set_printoptions(threshold=np.nan)

def split_list(l, n):
    list = []
    for j in range(0, len(l), n):
        if (j + n < len(l)):
            list.append(np.array(l[j:j + n]))
    return list

def get_songs(path):
    '''
    :param path: path to the songs directory
    :return: array of songs w/ timestamp events
    '''
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs

#Hyperparams
learning_rate = .1
#Batch Size
batch_size = 1000
#Number of training
epochs = 10
#number of features
num_features = 156
#Layers in the hidden lstm layer
layer_units = 156
#Number of time steps to use
n_steps = 10
#Song directore
songs = get_songs('./beeth')

#process songs and take timestamp cuts
input_sequence = []
expected_output = []

for song in songs:
    for offset in range(len(song) - n_steps - 1):
        input_sequence.append(song[offset:offset + n_steps])
        expected_output.append(song[offset + n_steps + 1])

batched_input = split_list(input_sequence, batch_size)
batched_expected_output = split_list(expected_output, batch_size)

#Weights biases and placeholders
w = tf.Variable(tf.truncated_normal([layer_units, num_features], stddev=.1))
b = tf.Variable(tf.truncated_normal([num_features], stddev=.1))

#x is examples by time by features
x = tf.placeholder(tf.float32, (None, 10, num_features))
#y is examples by examples by features
y = tf.placeholder(tf.float32, (None, num_features))

def RNN(x):
    '''
    :param x: rnn input data
    :return: rnn last timestamp output
    '''
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(layer_units)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    #must by transposed to time by examples by features from examples by time by features
    return tf.sigmoid(tf.matmul(tf.transpose(outputs, perm=[1, 0, 2])[-1], w) + b)

pred = RNN(x)

# Cross entropy loss
cost = tf.losses.softmax_cross_entropy(y, logits=pred)
#Train with Adam Optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.round(pred), y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #train for epoch epochs
    for i in tqdm(range(epochs)):
        for batch in range(len(batched_input)):
            sess.run(optimizer, feed_dict={x: batched_input[batch], y: batched_expected_output[batch]})
        print(sess.run(cost,feed_dict={x: input_sequence, y:expected_output}))
