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

def process_data(songs, n_steps):
    expected_output = []

    for song in tqdm(songs, desc="{0}.pad/seq".format(model_name), ascii=True):

        if (n_steps):
            song = split_list(song, n_steps)

        expected_output = expected_output + song

    seqlens = [n_steps for i in range(len(expected_output))]
    return expected_output, seqlens

model_name = 'C_RNN_GAN_V3_D1'
song_directory = './classical'
learning_rate_G = .005
lr = .01
discriminator_lr = .005
pretraining_epochs = 30
epochs = 100
load_from_saved = True
num_features = 4
layer_units = 200
max_n_steps = 100 # time steps
n_steps = 4
max_songs = 400
report_interval = 1
songs = midiprocess.get_songs(song_directory, model_name, max_songs)
expected_output, seqlens = process_data(songs, n_steps)
batch_size = int(sum(seqlens)/n_steps - 1)

lstm = LSTM(model_name, num_features, layer_units, batch_size, g_lr=learning_rate_G, d_lr=discriminator_lr, lr=lr)

lstm.start_sess(load_from_saved=load_from_saved)

lstm.trainLSTM(expected_output, epochs=pretraining_epochs, report_interval=1, time_steps=n_steps, seqlens=seqlens, batch_size=batch_size)

for i in tqdm(range(epochs)):
    expected_output, seqlens = process_data(songs, n_steps)
    batch_size = int(sum(seqlens) / n_steps - 1)
    lstm.trainAdversarially(expected_output, 10, report_interval=report_interval, seqlens=seqlens, batch_size=batch_size, time_steps=n_steps)
    if n_steps < max_n_steps:
        n_steps += 4

lstm.end_sess()



