from lstmClass import LSTM
import midi_manipulation
from tqdm import tqdm
import numpy as np

model_name = 'lstm_c02'
song_directory = './beeth'
learning_rate = .5
batch_size = 10
load_from_saved = True
epochs = 1
num_features = 156
layer_units = 156
n_steps = 100  # time steps
max_songs = 3
report_interval = 1

songs = midi_manipulation.get_songs(song_directory, model_name, max_songs)

input_sequence = []
expected_output = []
seqlens = []
max_seqlen = max(map(len, songs))

for song in tqdm(songs, desc="{0}.pad/seq".format(model_name)):
    seqlens.append(len(song) - 1)
    if (len(song) < max_seqlen):
        song = np.pad(song, pad_width=(((0, max_seqlen - len(song)), (0, 0))), mode='constant', constant_values=0)

    input_sequence.append(song[0:len(song) - 2])
    expected_output.append(song[1:len(song) - 1])

starter = np.transpose(input_sequence[:2][:10], (1, 0, 2))

lstm = LSTM(model_name, song_directory, num_features, layer_units, batch_size, load_from_saved, max_songs, learning_rate)

lstm.start_sess()

lstm.trainAdversarially(input_sequence, expected_output, epochs, report_interval=report_interval, seqlens=seqlens)

print(lstm.generateSequence(starter, 100))

lstm.end_sess()



