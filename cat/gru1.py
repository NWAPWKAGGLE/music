import tensorflow as tf
from tensorflow.contrib import rnn

layer_size = 512
vocab_size = 88
batch_size = 4
num_layers = 3

dropout = tf.placeholder(tf.float32)
weights = tf.Variable(tf.random_normal([layer_size, vocab_size]))
biases = tf.Variable(tf.random_normal([vocab_size]))

# batch_size, max_time, vocab_size

x = tf.placeholder(tf.float32, (1, None, vocab_size))
y = tf.placeholder(tf.float32, (1, 1, vocab_size))

cells = []
for i in range(num_layers):
    cell = rnn.GRUCell(layer_size)
    cell = rnn.DropoutWrapper(cell, output_keep_prob=1.0-dropout)
    cells.append(cell)
cell = rnn.MultiRNNCell(cells)

outputs, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

print(outputs)

