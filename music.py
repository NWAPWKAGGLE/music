from lstmClass import LSTM
import midi_manipulation
from tqdm import tqdm
import numpy as np

def split_list(l, n):
    list = []
    for j in range(0, len(l), n):
        if (j+n < len(l)):
            list.append(np.array(l[j:j+n]))
    return list

def process_data(songs, n_steps):
    expected_output = []
    seqlens = []
    max_seqlen = max(map(len, songs))

    for song in tqdm(songs, desc="{0}.pad/seq".format(model_name), ascii=True):
        if (n_steps):
            song = split_list(song, n_steps)

        expected_output = expected_output + song

    seqlens = [n_steps for i in range(len(expected_output))]
    return expected_output, seqlens

model_name = 'lstm_i04'

song_directory = './beeth'
learning_rate = .1
batch_size = 0
load_from_saved = False
epochs = 300
num_features = 156
layer_units = 156
n_steps = 10 # time steps
max_songs = None
report_interval = 1

songs = midi_manipulation.get_songs(song_directory, model_name, max_songs)

lstm = LSTM(model_name, num_features, layer_units, batch_size, learning_rate)

lstm.start_sess(load_from_saved=load_from_saved)

for j in range(40):
    expected_output, seqlens = process_data(songs, n_steps)
    lstm.trainAdversarially(expected_output, epochs, report_interval=report_interval, seqlens=seqlens)
    n_steps += 10

lstm.end_sess()



