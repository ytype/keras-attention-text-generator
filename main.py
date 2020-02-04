import argparse
from libs.processing import processing 
from libs.embedding import embedding
from libs.attention import attention_3d_block
from libs.getModel import get_model
from libs.generate import generate
from matplotlib import pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='keras attention text generator')

parser.add_argument('--file', required=True, help='train data file')
parser.add_argument('--seed', required=True, help='seed file')
parser.add_argument('--epoch', required=True, default=100, help='epoch')
parser.add_argument('--batch_size', required=False, default=64, help='batch_size')

args = parser.parse_args()

f = open(args.file,'r')
data = processing(f.read())
f.close()

x,y = embedding(data)

n = 300000

m = get_model()
m.compile(optimizer='adam', loss='categorical_crossentropy')
    
history  = m.fit(x, y, epochs=args.epoch, batch_size=args.batch_size, validation_split=0)
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('img/loss.png')

generate()
