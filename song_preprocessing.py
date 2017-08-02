from midi_manipulation import midiToNoteStateMatrix
from glob import glob
from tqdm import tqdm
import numpy as np

def get_songs(path):
    '''
    :param path: path to the songs directory
    :return: array of songs w/ timestamp events
    '''
    files = glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files, desc="loading songs"):
        try:
            song = np.array(midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs

def seq_songs(songs, n_steps):
    input_sequence = []
    expected_output = []

    for song in songs:
        for offset in range(len(song) - n_steps - 1):
            input_sequence.append(song[offset:offset + n_steps])
            expected_output.append(song[offset + n_steps + 1])

    return input_sequence, expected_output