import tensorflow as tf
import glob
from tqdm import tqdm
import numpy as np
from midi_manipulation import midiToNoteStateMatrix

def split_list(l, n):
    list = []
    for j in range(0, len(l), n):
        if (j+n < len(l)):
            list.append(np.array(l[j:j+n]))
    return list

def get_songs(path):
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
learning_rate = .05
layer_units = 127
batch_size = 100
n_steps = 100
songs = get_songs('./beeth')
notes_in_dataset = []
for i in range(len(songs)):
    notes_in_dataset.append(split_list(songs[i], batch_size))


initial_state = state = tf.zeros([batch_size, lstm.state_size])
probabilities = []
loss = 0.0
w = tf.Variable('softmax_w', tf.random_normal([layer_units, 127]))
b = tf.Variable('softmax_b', tf.random_normal([127]))
x = tf.placeholder([None, 100, 127])
y = tf.placeholder([None, 127])

def RNN(x, w, b):

    x = tf.unstack(x, n_steps, 1)

    lstm_cell = tf.contrib.dynamic_rnn.BasicLSTMCell(layer_units)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.sigmoid(tf.matmul(outputs[-1], w) + b)

pred = RNN(x, w, b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.round(pred))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
