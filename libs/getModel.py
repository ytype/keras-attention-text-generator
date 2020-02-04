import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keract import get_activations
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import plot_model
from libs.attention import attention_3d_block

INPUT_DIM = 100
TIME_STEPS = 20

def get_model(x):
    #inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    inputs = Input(shape=(x.shape[1], x.shape[2]))
    rnn_out = LSTM(32, return_sequences=True)(inputs)
    dropout_output = Dropout(0.2)(rnn_out)
    attention_output = attention_3d_block(dropout_output)
    output = Dense(y.shape[1], activation='softmax', name='output')(attention_output)
    m = Model(inputs=[inputs], outputs=[output])
    print(m.summary())
    plot_model(m, to_file='img/model.png')
    return m
