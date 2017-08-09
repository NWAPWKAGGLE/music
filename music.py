from lstmClass import LSTM
import midi_manipulation
from tqdm import tqdm
import numpy as np

model_name = 'lstm_h02'

song_directory = './beeth'
learning_rate = .1
batch_size = 10
load_from_saved = True
epochs = 0
num_features = 156
layer_units = 156
n_steps = 30  # time steps
max_songs = 1
report_interval = 10

songs = midi_manipulation.get_songs(song_directory, model_name, max_songs)

expected_output = []
seqlens = []
max_seqlen = max(map(len, songs))

for song in tqdm(songs, desc="{0}.pad/seq".format(model_name)):
    seqlens.append(len(song) - 1)
    if (len(song) < max_seqlen):
        song = np.pad(song, pad_width=(((0, max_seqlen - len(song)), (0, 0))), mode='constant', constant_values=0)

    expected_output.append(song[1:n_steps])

lstm = LSTM(model_name, num_features, layer_units, batch_size, learning_rate)

lstm.start_sess(load_from_saved=load_from_saved)

lstm.trainAdversarially(expected_output, epochs, report_interval=report_interval, seqlens=seqlens)

sequences = lstm.generate_sequence(1, 30)
#print(sequences)
lstm.generate_midi_from_sequences(sequences, './musicgenerated/')

lstm.end_sess()



