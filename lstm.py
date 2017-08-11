from adversarial import AdversarialNet
import midi_manipulation
from tqdm import tqdm
import numpy as np

def split_list(l, n):
    list_ = []
    for j in range(0, len(l), n):
        if (j+n < len(l)):
            list_.append(np.array(l[j:j+n]))
    return list_

def process_data(songs_, n_steps_):
    expected_output = []
    min_seqlen = min(map(len, songs_))
    if min_seqlen < n_steps_:
        n_steps_ = min_seqlen

    for song in tqdm(songs_, desc="{0}.pad/seq".format(model_name), ascii=True):
        if (n_steps_):
            song = split_list(song, n_steps_)

        expected_output = expected_output + song

    seqlens = [n_steps_ for i in range(len(expected_output))]
    return expected_output, seqlens

model_name = 'adv_a01'

song_directory = './beeth'
max_songs = 3

batch_size = 0
epochs = 1
num_features = 156
layer_units = 156
num_layers = 2
time_steps = 10

report_interval = 1

songs = midi_manipulation.get_songs(song_directory, model_name, max_songs)
expected_output, seqlens = process_data(songs, time_steps)

with AdversarialNet.new(model_name, num_features, layer_units, num_layers) as net:
    tqdm.write('############# MODEL IS {0}TRAINED #############'.format('' if net.trained else 'UN'))
    net.learn_multiple_epochs(expected_output, seqlens, g_learning_rate=0.01, d_learning_rate=0.007, epochs=1, report_interval=1)