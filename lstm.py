import tensorflow as tf
import glob
from tqdm import tqdm
import numpy as np
from midi_manipulation import midiToNoteStateMatrix

def split_list(l, n):
    list = []
    for j in range(0, len(l), n):
        if (j+n < len(l)):
            list.append(np.array(l[j:j+n]))
    return list

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs

#Hyperparams
learning_rate = .05
layer_units = 10
batch_size = 100

songs = get_songs('./beeth')
notes_in_dataset = []
for i in range(len(songs)):
    print(split_list(songs[i], batch_size))
    notes_in_dataset.append(split_list(songs[i], batch_size))



