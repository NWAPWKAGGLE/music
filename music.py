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

model_name = 'lstm_i01'

song_directory = './beeth'
learning_rate = .1
batch_size = 10
load_from_saved = True
epochs = 10
num_features = 156
layer_units = 156
n_steps = 10 # time steps
max_songs = None
report_interval = 1


songs = midi_manipulation.get_songs(song_directory, model_name, max_songs)

expected_output = []
seqlens = []
max_seqlen = max(map(len, songs))

for song in tqdm(songs, desc="{0}.pad/seq".format(model_name)):

    if (n_steps):
        song = split_list(song, n_steps)
    else:
        seqlens.append(len(song) - 1)
        if (len(song) < max_seqlen):
            song = np.pad(song, pad_width=(((0, max_seqlen - len(song)), (0, 0))), mode='constant', constant_values=0)

    expected_output = expected_output + song

seqlens = [n_steps for i in range(len(expected_output))]

lstm = LSTM(model_name, num_features, layer_units, batch_size, learning_rate)

lstm.start_sess(load_from_saved=load_from_saved)

lstm.trainAdversarially(expected_output, epochs, report_interval=report_interval, seqlens=seqlens, batch_size=100)

lstm.end_sess()



