import tensorflow as tf
from midi import noteStateMatrixToMidi



#Hyperparams
learning_rate = .05
layer_units = 10
batch_size = 20
lstm = tf.contrib.rnn.BasicLSTMCell(layer_units)
state = tf.zeros([batch_size, layer_units])

for current_batch_of_notes in notes_in_dataset







