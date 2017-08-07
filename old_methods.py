def _error(self, xseq, yseq, seqlens):
    if not self.managed:
        raise RuntimeError("TFRunner must be in with statement")
    else:
        return self.sess.run('cost:0', feed_dict={'x:0': xseq, 'y:0': yseq, 'seq_lens:0': seqlens})


def validate(self, xseq, yseq, seqlens):
    # TODO: Meaningful???
    if not self.trained:
        raise RuntimeError("attempted to call validate() on untrained model")
    else:
        return self._error(xseq, yseq, seqlens)


def generate_music_sequences_from_noise(self, num_timesteps, num_songs):
    if not self.managed:
        raise RuntimeError("TFRunner must be in with statement")
    else:
        if not self.trained:
            raise RuntimeError("attempted to call _generate_music() on untrained model")
        else:
            xseq = tf.truncated_normal((num_songs, num_timesteps, 156), mean=.5, stddev=.1)
            seqlens = np.empty([num_songs])
            seqlens.fill(num_timesteps)
            return self.sess.run('y_', feed_dict={'x:0': xseq.eval(session=self.sess), 'seq_lens:0': seqlens})


'''def generate_music_sequences_recursively(self, num_timesteps, num_songs, starter, starter_length, layer_units):
    if not self.managed:
        raise RuntimeError("TFRunner must be in with statement")
    else:
        if not self.trained:
            raise RuntimeError("attempted to call generate_music_sequences_recursively() on untrained model")
        else:
            #input_ = tf.cast(starter, dtype=np.float32)[-1]
            #input_ = starter = t
            with tf.variable_scope('lstm_layer1/rnn', reuse=True):
                outputs = [tf.zeros([1, 156]),]
                state = self._cell.zero_state(1, tf.float32)
                for _ in tqdm(range(num_timesteps)):
                    o, s = self._cell(outputs[-1], state)
                    state = s
                    outputs.append(o)
                return self.sess.run(outputs[-1])'''

'''def generate_music_sequences_recursively(self, num_timesteps, num_songs, starter, starter_length, layer_units):
    if not self.managed:
        raise RuntimeError("TFRunner must be in with statement")
    else:
        if not self.trained:
            raise RuntimeError("attempted to call generate_music_sequences_recursively() on untrained model")
        else:
            sequence = tf.cast(starter, dtype=tf.float32)
            thesestates = None

            with tf.variable_scope('lstm_layer1', reuse=True, custom_getter=custom_getter):
                sequence, thesestates = tf.nn.dynamic_rnn(self._cell, sequence, dtype=tf.float32,
                                                          initial_state=thesestates)

            sequence = tf.expand_dims(sequence[-1], 0)
            # TODO: Multiply by weights and add bias?

            outputs = starter
            for i in tqdm(range(num_timesteps)):
                np.append(outputs, np.squeeze(self.sess.run(sequence)))
            return np.transpose(outputs, (1, 0, 2))
'''


def generate_midi_from_sequences(self, sequence, dir_path):
    for i in range(len(sequence)):
        print(sequence[i])
        mm.noteStateMatrixToMidi(sequence[i], dir_path + 'generated_chord_{}'.format(i))