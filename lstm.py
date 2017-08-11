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
learning_rate_G = .06
#learning_rate_D = .01
batch_size = 0
epochs = 5
num_features = 156
layer_units = 156
n_steps = 50 # time steps
max_songs = None
report_interval = 4

songs = midi_manipulation.get_songs(song_directory, model_name, max_songs)





with AdversarialNet.load_or_new(model_name, learning_rate, num_features, layer_units, num_layers) as net:
    tqdm.write('############# MODEL IS {0}TRAINED #############'.format('' if net.trained else 'UN'))
    net.learn(input_sequence, seqlens, epochs=3, report_interval=1)