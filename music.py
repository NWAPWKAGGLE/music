from lstmClass import LSTM
import midi_manipulation
from tqdm import tqdm
import numpy as np

model_name = 'lstm_d01'

song_directory = './beeth'
learning_rate = .5
batch_size = 10
load_from_saved = False
epochs = 10
num_features = 156
layer_units = 156
n_steps = 100  # time steps
max_songs = 6
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

starter = np.transpose(input_sequence[1:2], (1, 0, 2))[0:10:1]

lstm = LSTM(model_name, num_features, layer_units, batch_size, learning_rate)

lstm.start_sess(load_from_saved=load_from_saved)

#lstm.trainLSTM(input_sequence, expected_output, epochs, report_interval=report_interval, seqlens=seqlens)

sequences = lstm.generate_sequence(starter, 100)
print(sequences)
lstm.generate_midi_from_sequences(sequences, './musicgenerated/')

lstm.end_sess()



