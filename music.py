from lstmClass import LSTM
import midiprocess

from tqdm import tqdm
import numpy as np

def split_list(l, n):
    list = []
    for j in range(0, len(l), n):
        if (j+n < len(l)):
            list.append(np.array(l[j:j+n]))
    return list

def process_data(songs, max_len):

    expected_output = []
    seqlens = []
    for song in songs:
        if (len(song) < max_len):
            seqlens.append(len(song))
            song = np.pad(song, pad_width=(((0, max_len- len(song)), (0, 0))), mode='constant', constant_values=0)
            expected_output.append(song)


    return expected_output, seqlens

model_name = 'NC_RNN_GAN_A1'
song_directory = './classical'
learning_rate_G = .01
lr = .01
#learning_rate_D = .01
batch_size = 2
load_from_saved = False
epochs = 300
num_features = 5
layer_units = 10
discriminator_lr = .01
n_steps = 100 # time steps
max_songs = 10
report_interval = 1


songs = midiprocess.get_songs(song_directory, model_name, max_songs)

lstm = LSTM(model_name, num_features, layer_units, batch_size, g_lr=learning_rate_G, d_lr=discriminator_lr, lr=lr, max_song_len=2000)

lstm.start_sess(load_from_saved=load_from_saved)

expected_output, seqlens = process_data(songs, 2000)

lstm.trainAdversarially(expected_output, epochs, report_interval=report_interval, seqlens=seqlens, batch_size=batch_size, max_song_len=5000)

lstm.end_sess()

