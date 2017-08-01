import tensorflow as tf
import glob
from tqdm import tqdm
import numpy as np
from midi_manipulation import midiToNoteStateMatrix
import os.path
from glob import iglob
from datetime import datetime

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
learning_rate = .05
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

save_dir = './saved_models'

report_interval = epochs / 10

model_name = 'lstm_a01'

def load_or_build(model_name, learning_rate, num_features, layer_units, num_steps, file_name=None):
    try:
        return load(model_name, file_name)
    except FileNotFoundError:
        return build(learning_rate, num_features, layer_units, num_steps)

def build(learning_rate, num_features, layer_units, num_steps):
    w = tf.Variable(tf.truncated_normal([layer_units, num_features], stddev=.1), name='w')
    b = tf.Variable(tf.truncated_normal([num_features], stddev=.1), name='b')

    #x is examples by time by features
    x = tf.placeholder(tf.float32, (None, num_steps, num_features), name='x')
    #y is examples by examples by features
    y = tf.placeholder(tf.float32, (None, num_features), name='x')

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(layer_units)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    pred = tf.sigmoid(tf.matmul(tf.transpose(outputs, perm=[1, 0, 2])[-1], w) + b, name='y_')

    cost = tf.identity(tf.losses.softmax_cross_entropy(y, logits=pred), name='cost')

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(cost)

    correct_pred = tf.equal(tf.round(pred), y, name='correct')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    return sess, saver


def load(model_name, file_name=None):
    if file_name is None:
        try:
            selector = os.path.join(save_dir, model_name, '*.ckpt.meta')
            file_name = max(iglob(selector), key=os.path.getctime)
        except ValueError:
            raise FileNotFoundError("This model does not exist...")
    else:
        file_name = os.path.join(save_dir, file_name)
        if not os.path.isfile(file_name):
            raise FileNotFoundError("Model file by name {0} does not exist...".format(model_name))
    sess = tf.Session()
    saver = tf.train.import_meta_graph(file_name)
    saver.restore(sess, file_name[:-5])
    return sess, saver


def save(sess, saver, model_name, err, i, epochs):
    s_path = os.path.join(save_dir, model_name, '{0}__{1}_{2}__{3}.ckpt'.format(err, i, epochs,
                                                                                str(datetime.now()).replace(':', '_')))
    return saver.save(sess, s_path)


with build(learning_rate, num_features, layer_units, n_steps) as state:
    sess, saver = state

    input_sequence = []
    expected_output = []

    for song in songs:
        for offset in range(len(song) - n_steps - 1):
            input_sequence.append(song[offset:offset + n_steps])
            expected_output.append(song[offset + n_steps + 1])
    for i in tqdm(range(epochs)):
        sess.run('optimizer', feed_dict={'x:0': input_sequence, 'y:0': expected_output})
        if i % report_interval == 0:
            print(sess.run('cost',feed_dict={'x:0': input_sequence, 'y:0': expected_output}))
