import glob
from tqdm import tqdm
import numpy as np

import mido

def get_song(path):
    '''

    :param path: path to the song
    :return: an array of midi bytes for the song

    '''
    mid = mido.MidiFile(path)
    song = []

    for msg in mid:
        if not (msg.is_meta or msg.type != 'note_on'):
            note_array = [msg.note, msg.velocity, msg.time]
            song.append(note_array)
    return song

def get_songs(path, model_name, max=None):
    '''
    :param path: path to the songs directory
    :return: array of songs w/ timestamp events
    '''
    files = glob.glob('{}/*.mid*'.format(path))
    files = files[:max] if max is not None else files

    songs = []
    c = 0
    for f in tqdm(files):
        songs.append(np.array(get_song(f)))
    return songs

def save_to_midi_file(song_array, name):
    mid = mido.MidiFile()
    track = mido.MidiTrack()

    mid.tracks.append(track)
    for i in range(len(song_array)):

        for j in range(3):
            if song_array[i][j] > 127:
                song_array[i][j] = 127

        track.append(mido.Message('note_on', note=song_array[i][0], velocity=song_array[i][1], time=song_array[i][2]))

    mid.save(name)
