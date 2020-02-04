from keras.utils import np_utils
import numpy as np

def embedding(str):
    chars = sorted(list(set(str)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    n_chars = len(str)
    n_vocab = len(chars)
    print ("Total Characters: ", n_chars)
    print ("Total Vocab: ", n_vocab)
    seq_length = 100
    dataX = []
    dataY = []

    for i in range(0, n_chars - seq_length, 1):
        seq_in = str[i:i + seq_length]
        seq_out = str[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print ("Total Patterns: ", n_patterns)

    x = np.reshape(dataX, (n_patterns, seq_length, 1))
    x = x / float(n_vocab)
    y = np_utils.to_categorical(dataY)
    return x,y