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
    min_seqlen = min(map(len, songs))
    if min_seqlen < n_steps:
        n_steps = min_seqlen

    for song in tqdm(songs, desc="{0}.pad/seq".format(model_name), ascii=True):
        if (n_steps):
            song = split_list(song, n_steps)

        expected_output = expected_output + song

    seqlens = [n_steps for i in range(len(expected_output))]
    return expected_output, seqlens

model_name = 'C_RNN_GAN_C1'

song_directory = './beeth'
learning_rate_G = .09
#learning_rate_D = .01
batch_size = 0
load_from_saved = False
epochs = 5
num_features = 156
layer_units = 156
n_steps = 50 # time steps
max_songs = 30
report_interval = 4

songs = midi_manipulation.get_songs(song_directory, model_name, max_songs)

lstm = LSTM(model_name, num_features, layer_units, batch_size, learning_rate_G)

lstm.start_sess(load_from_saved=load_from_saved)

for j in range(50):

    expected_output, seqlens = process_data(songs, n_steps)

    lstm.trainAdversarially(expected_output, epochs, report_interval=report_interval, seqlens=seqlens)
    n_steps += 50

lstm.end_sess()



