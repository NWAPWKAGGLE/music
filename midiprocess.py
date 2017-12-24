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
    tempo = 500000
    ticks_per_beat = mid.ticks_per_beat
    print(ticks_per_beat)
    for msg in mid:
        if msg.type=='set_tempo':
            tempo = msg.tempo

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

    mid.tracks.append(track)
    for i in range(len(song_array)):
        for j in range(3):
            if song_array[i][j] < 0:
                song_array[i][j] = 0
            elif song_array[i][j] > 127:
                song_array[i][j] = 127
        time = int(round(mido.second2tick(song_array[i][2], mid.ticks_per_beat, 500000)))

        track.append(mido.Message('note_on', note=int(song_array[i][0]), velocity=int(song_array[i][1]), time=time))
    mid.save(name)

def convert_timestamps_to_notes(song):

    new_song = []
    #iterate through the song
    for i in range(len(song)):

        #check if it is the start of a note
        if (song[i][1] != 0):

            note_length = song[i][2]
            time_till_next_note = 0
            for j in range(len(song)-i-1):

                k=i+j+1
                note_length += song[k][2]
                #check if it is the note off event
                if (song[i][0] == song[k][0]) and (song[k][1] == 0):

                    #if the next note is the note off event, set the time to wait to the length of the note + the time after note off event
                    if j==0:
                        time_till_next_note = song[i+1][2]

                    #otherwise, it's just the time after the current note
                    else:
                        time_till_next_note = song[i][2]
                    break

            new_song.append([song[i][0], song[i][1], note_length, time_till_next_note])

    return new_song

def convert_notes_to_timestamps(song):
    print(song)
    #keep track of how much time there is before a note ends
    time_till_note_end = [song[i][2] for i in range(len(song))]

    new_song = []

    #loop through the song
    for i in tqdm(range(len(song))):

        #array of notes that are ending before the next note starts
        ending_notes = []

        for j in range (i):
            if (time_till_note_end[j] >= 0) and (time_till_note_end[j] < song[i][3]):
                if (song[j][0] == 61):
                    print(i)
                    print(time_till_note_end[j])
                ending_notes.append([j, time_till_note_end[j]])

        ending_notes = sorted(ending_notes, key=lambda x: x[1])

        for k in range(len(ending_notes)):
            if (k != len(ending_notes)-1):
                new_song.append([song[ending_notes[k][0]][0], 0, time_till_note_end[ending_notes[k+1][0]]-time_till_note_end[ending_notes[k][0]]])
            else:
                new_song.append([song[ending_notes[k][0]][0], 0, time_till_note_end[ending_notes[k][0]]])

        for h in range(i+1):
            current_time = time_till_note_end[h]
            time_till_note_end[h] -= song[i][3]

        if (song[i][2] > song[i][3]):
            new_song.append([song[i][0], song[i][1], song[i][3]])
        else:
            new_song.append([song[i][0], song[i][1], song[i][2]-song[i][3]])

    return new_song

test_song = get_song('./classical/mond_1.mid')
converted_song = convert_timestamps_to_notes(test_song)
original = convert_notes_to_timestamps(converted_song)
for i in range(10):
    print(test_song[i])
    print(original[i])
save_to_midi_file(converted_song, "./mond_1.mid")