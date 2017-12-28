import glob
from tqdm import tqdm
import numpy as np

import mido
test_song = [[30, 33, 0],
             [40, 33, 1],
             [30, 0, 0],
             [40, 0, 0]]
def get_song(path):
    '''
    :param path: path to the song
    :return: an array of midi bytes for the song
    '''
    mid = mido.MidiFile(path)
    song = []

    time_between = 0
    for msg in mid:

        if not (msg.is_meta or msg.type != 'note_on'):
            note_array = [msg.note, msg.velocity, round(msg.time+time_between, 5)]
            song.append(note_array)
            time_between = 0
        else:
            time_between += msg.time
    return song

def get_songs(path, model_name, max=None):
    '''
    :param path: path to the songs directory
    :return: array of songs w/ timestamp events
    '''
    files = glob.glob('{}/*.mid*'.format(path))
    files = files[:max] if max is not None else files

    songs = []

    for f in tqdm(files):
        songs.append(np.array(convert_timestamps_to_notes(get_song(f))))
    return songs

def save_to_midi_file(song_array, name):
    for i in range(len(song_array)):
        for j in range(4):
            if song_array[i][j] < 0:
                song_array[i][j] = 0

    song_array = convert_notes_to_timestamps(song_array)
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    tempo = 500000
    mid.tracks.append(track)

    for i in range(len(song_array)):
        if (song_array[i][0] == -1):
            #track.append(mido.Message(type='set_tempo', tempo=song_array[i][1]))
            tempo = song_array[i][1]
            continue

        for j in range(3):
            if song_array[i][j] < 0:
                song_array[i][j] = 0
            elif song_array[i][j] > 127:
                song_array[i][j] = 127
        time = int(round(mido.second2tick(song_array[i][2], mid.ticks_per_beat, tempo)))

        track.append(mido.Message('note_on', note=int(song_array[i][0]), velocity=int(song_array[i][1]), time=time))
    mid.save(name)

def convert_timestamps_to_notes(song):

    new_song = []
    #iterate through the song
    for i in range(len(song)):

        #check if it is the start of a note

        if (song[i][1] != 0):
            time_till_next_note_on = 0

            note_length = 0

            for j in range(len(song)-i-1):

                k=i+j+1
                time_till_next_note_on += song[k][2]
                if (song[k][1] != 0):
                    break

            for j in range(len(song)-i-1):

                k=i+j+1

                note_length += song[k][2]

                #check if it is the note off event
                if (song[i][0] == song[k][0]) and (song[k][1] == 0):
                    break

            new_song.append([song[i][0], song[i][1], round(note_length, 5), round(time_till_next_note_on, 5)])

    return new_song

def convert_notes_to_timestamps(song):

    #keep track of how much time there is before a note ends
    time_till_note_end = [song[i][2] for i in range(len(song))]
    time_from_last_note = 0
    new_song = []
    time_till_next_note = 0
    #loop through the song
    for i in tqdm(range(len(song))):

        #array of notes that are ending before the next note starts
        if (song[i][2] == 0):
            new_song.append([song[i][0], song[i][1], 0])
            new_song.append([song[i][0], 0, 0])
            continue



        new_song.append([song[i][0], song[i][1], time_from_last_note])

        ending_notes = []

        time_from_last_note = song[i][3]
        for j in range(i+1):
            if (i < len(song)-2):
                if (round(time_till_note_end[j], 5) > 0) and (round(time_till_note_end[j], 5) <= round(song[i][3], 5)):
                    ending_notes.append([j, time_till_note_end[j]])

        ending_notes = sorted(ending_notes, key=lambda x: x[1])
        if (len(ending_notes) != 0):
            time_from_last_note -= ending_notes[-1][1]

        for k in range(len(ending_notes)):
            if (k != len(ending_notes) - 1):
                new_song.append([song[ending_notes[k][0]][0], 0,
                                 ending_notes[k][1]])
            else:
                new_song.append([song[ending_notes[k][0]][0], 0, ending_notes[k][1]])

        for h in range(i+1):
            time_till_note_end[h] -= song[i][3]

    return new_song