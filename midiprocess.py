import glob
from tqdm import tqdm
import numpy as np
import os
import mido

def get_song(path):
    '''

    :param path: path to the song
    :return: an array of midi bytes for the song

    '''
    mid = mido.MidiFile(path)


    song = []
    tempo = 500000
    ticks_per_beat = mid.ticks_per_beat

    for msg in mid:
        if msg.type=='set_tempo':

            tempo = msg.tempo
        if not (msg.is_meta or msg.type != 'note_on'):
            note_array = [msg.note, msg.velocity, mido.second2tick(msg.time, ticks_per_beat, tempo)]
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
        try:
            song = get_song(f)
        except:
            os.remove(f)
            continue

        songs.append(np.array(convert_timestamps_to_notes(song)))
    return songs

def save_to_midi_file(song_array, name):
    mid = mido.MidiFile()
    track = mido.MidiTrack()

    mid.tracks.append(track)
    for i in range(len(song_array)):
        for j in range(3):
            if song_array[i][j] > 127:
                song_array[i][j] = 127
            elif song_array[i][j] < 0:
                song_array[i][j] = 0
        track.append(mido.Message('note_on', note=int(song_array[i][0]), velocity=int(song_array[i][1]), time=int(round(song_array[i][2], 0))))

    mid.save(name)

def convert_timestamps_to_notes(song):

    new_song = []
    #iterate through the song
    for i in range(len(song)):

        #check if it is the start of a note
        if (song[i][1] != 0):

            note_length = 0
            time_till_next_note = song[i][2]
            for j in range(len(song)-i-1):

                k=i+j+1
                note_length += song[k][2]

                #check if it is the note off event
                if (song[i][0] == song[k][0]) and (song[k][1] == 0):

                    #if the next note is the note off event, set the time to wait to the length of the note + the time after note off event
                    if j==1:
                        time_till_next_note = song[i+1][2]+song[i][2]
                    #otherwise, it's just the time after the current note
                    break

            new_song.append([song[i][0], song[i][1], note_length, time_till_next_note])

    return new_song

def convert_notes_to_timestamps(song):

    #keep track of how much time there is before a note ends
    time_till_note_end = [song[i][2] for i in range(len(song))]
    print(time_till_note_end)
    new_song = []

    #loop through the song
    for i in tqdm(range(len(song))):

        #array of notes that are ending before the next note starts
        ending_notes = []

        for j in range(4):
            if song[i][j] > 127:
                song[i][j] = 127
            elif song[i][j] < 0:
                song[i][j] = 0

        for j in range (i+1):

            if (time_till_note_end[j] > 0) and (time_till_note_end[j] <= song[i][3]):
                ending_notes.append([j, time_till_note_end[j]])

        ending_notes = sorted(ending_notes, key= lambda x: x[1])

        for k in range(len(ending_notes)):
            if (k != len(ending_notes)-1):
                new_song.append([song[ending_notes[k][0]][0], 0, time_till_note_end[ending_notes[k+1][0]]-time_till_note_end[ending_notes[k][0]]])
            else:
                new_song.append([song[ending_notes[k][0]][0], 0, song[i][3]-time_till_note_end[ending_notes[k][0]]])

        for h in range(i):
            time_till_note_end[h] -= song[i][3]

        if (song[i][2] > song[i][3]):
            new_song.append([song[i][0], song[i][1], song[i][3]])
        else:
            new_song.append([song[i][0], song[i][1], song[i][2]])

    return new_song


