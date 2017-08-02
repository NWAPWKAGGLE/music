import song_preprocessing as sp
from jrstnets import LSTMNetFactory
import os
from tqdm import tqdm

print(os.getcwd())
print(os.path.isdir('./model_saves'))

# HYPERPARAMS
learning_rate = .05
epochs = 10
num_features = 156
lstm_layer_size = 156
time_steps = 10
model_name = 'lstm_a01'
song_dir = './beeth'

input_sequence, expected_output = sp.seq_songs(sp.get_songs(song_dir), time_steps)

with LSTMNetFactory.load_or_new(model_name, learning_rate, num_features, lstm_layer_size, time_steps) as net:
    tqdm.write(str(net.trained))
    net.learn(input_sequence, expected_output, epochs, report_interval=1)
