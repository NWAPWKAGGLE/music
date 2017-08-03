import numpy as np
from jrstnets import LSTMNetFactory





#generates an array of 156 column note lists with rows equal to note_number+1 using a note_list that is passed in
def song_create(note_number, note_list):
    music_array = np.array(note_list)

    with LSTMNetFactory.load_or_new('lstm_a01', .05, 156, 156, 10) as net:
        for x in range(note_number):
            note_list= (net.feed_forward(note_list))
            music_array = np.append(music_array,note_list)


test= np.zeros(0,10,156)
song_create(100, test)