import os
import sys
import numpy as np
import pickle as pkl
import h5py
import scipy.io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(1337) # for reproducibility

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
# from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.wrappers import Bidirectional

print('Building and compiling model...')

seq_input = Input(shape=(1000, 4))
H = Conv1D(320, 26, padding='valid', input_shape=(1000,4), activation='relu')(seq_input)
H = MaxPooling1D(pool_size=13, strides=13)(H)
H = Dropout(0.2)(H)

H = Bidirectional(LSTM(320, input_shape=(None, 320), return_sequences=True))(H)
H = Bidirectional(LSTM(320, input_shape=(None, 320), return_sequences=True))(H)
H = Dropout(0.5)(H)
H = Flatten()(H)

H = Dense(input_dim=75*320, units=925, activation='relu')(H)
H = Dense(input_dim=925, units=100, activation='sigmoid')(H)

track_input = Input(shape=(1000,))
G = Dense(input_shape=(1000,), units=250, activation='relu')(track_input)
G = Dense(units=100, activation='sigmoid')(G)

fc_input = concatenate([H,G])

F = Dense(input_shape=(200,), units = 6, activation='sigmoid')(fc_input)

model = Model(inputs=[seq_input, track_input], outputs=[F])
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

model.summary()

data = scipy.io.loadmat('F:\\data\\subc_seqs_no19.fa.data.1.mat')
hist_data = pkl.load(open('F:\\data\\H3K36me3\\GM12878_H3K36me3_tracks.pkl', 'rb'))

seqs = data['seqs']
labels = data['labels']
indices = data['indices'][0]
stringData = data['stringData']
res = hist_data['res']
mean_tracks = hist_data['mean']

seqs = seqs[:,49000:50000,:]

seg_length = int(100000/res)

del hist_data

training_tracks = []

for i, str in enumerate(stringData):
	chrm = str.split()[0]
	start = int(int(str.split()[1]) / res)
	end = int(int(str.split()[2]) / res)
	track = mean_tracks[chrm]['means'][start + int(indices[i] / 100000) * seg_length:start + (int(indices[i] / 100000) + 1) * seg_length]

	training_tracks.append(track)

training_tracks = np.asarray(training_tracks)
#training_tracks = np.expand_dims(training_tracks, axis=2)
print(training_tracks.shape, seqs.shape, labels.shape)

model.fit(x=[seqs, training_tracks], y=labels, epochs=25, batch_size=64, validation_split=0.2)
print(model.predict(x=[seqs, training_tracks]))