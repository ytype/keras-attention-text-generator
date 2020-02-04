import argparse
import sys
from libs.processing import processing 
from libs.embedding import embedding
from libs.getModel import get_model
from matplotlib import pyplot as plt
from keract import get_activations
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='keras attention text generator')

parser.add_argument('--file', required=True, help='train data file')
parser.add_argument('--epoch', required=True, default=100, help='epoch')
parser.add_argument('--batch_size', required=False, default=64, help='batch_size')

args = parser.parse_args()

f = open(args.file,'r')
data = processing(f.read())
f.close()

x,y,chars,n_vocab,dataX,seq_length = embedding(data)

n = 300000

m = get_model(x,y)
m.compile(optimizer='adam', loss='categorical_crossentropy')

history  = m.fit(x, y, epochs=int(args.epoch), batch_size=int(args.batch_size), validation_split=0)
import sys
int_to_char = dict((i, c) for i, c in enumerate(chars))

start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(100):
    out_x = np.reshape(pattern, (1, len(pattern), 1))
    out_x = out_x / float(n_vocab)
    prediction = m.predict(out_x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")


num_simulations = x.shape[2]
attention_vectors = np.zeros(shape=(num_simulations, seq_length))
for i in range(num_simulations):
    #testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
    activations = get_activations(m, x, layer_name='attention_weight')
    #activations = K.function([m.layers[0].input], [m.layers[1].output])
    attention_vec = np.mean(activations['attention_weight'], axis=0).squeeze()
    assert np.abs(np.sum(attention_vec) - 1.0) < 1e-5
    attention_vectors[i] = attention_vec
print("attention vector: ",attention_vectors)