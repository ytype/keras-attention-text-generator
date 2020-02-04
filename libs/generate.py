import sys
from matplotlib import pyplot as plt

def generate():
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

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
    attention_df = pd.DataFrame(attention_vec, columns=['attention (%)'])
    attention_df.plot(kind='bar', figsize=(50, 6), title='Attention Mechanism as a function of input dimensions.')
    plt.savefig('attention.png')
