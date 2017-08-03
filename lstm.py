import tensorflow as tf
import glob
from tqdm import tqdm
import numpy as np
from midi_manipulation import midiToNoteStateMatrix
from midi_manipulation import noteStateMatrixToMidi
from jrstnets import LSTMNetFactory

#np.set_printoptions(threshold=np.nan)  # TODO: should be np.inf??

model_name = 'lstm_a02'
song_directory = './beeth'
learning_rate = .5
batch_size = 10
epochs = 10
num_features = 156
layer_units = 156
n_steps = 100  # time steps



######################## PREPROCESSING ############################

def get_songs(path):
    '''
    :param path: path to the songs directory
    :return: array of songs w/ timestamp events
    '''
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files, desc='{0}.get_songs({1})'.format(model_name, path)):
        try:

            song = np.array(midiToNoteStateMatrix(f))
            songs.append(song)

        except Exception as e:
            raise e
    return songs
songs = get_songs(song_directory)

input_sequence = []
expected_output = []
seqlens = []
max_seqlen = max(map(len, songs))

for song in tqdm(songs, desc="{0}.pad/seq".format(model_name)):
    seqlens.append(len(song)-1)
    if (len(song) < max_seqlen):

        song = np.pad(song, pad_width=(((0, max_seqlen-len(song)), (0,0))), mode='constant', constant_values=0)

    input_sequence.append(song[0:len(song)-2])
    expected_output.append(song[1:len(song)-1])

############################## END PREPROCESSING ############################



with LSTMNetFactory.load_or_new(model_name, learning_rate, num_features, layer_units, max_seqlen-2) as net:
    tqdm.write(str(net.trained))
    net.learn(input_sequence, expected_output, seqlens, epochs, report_interval=1)