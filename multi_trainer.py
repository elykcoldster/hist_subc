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

def loadmats(mat_files):
	seqs = None
	labels = None
	indices = None
	stringData = None
	for mat_file in mat_files:
		data = scipy.io.loadmat(mat_file)
		if seqs is None:
			seqs = data['seqs']
			labels = data['labels']
			indices = data['indices'][0]
			stringData = data['stringData']
		else:
			seqs = np.concatenate((seqs, data['seqs']))
			labels = np.concatenate((labels, data['labels']))
			indices = np.concatenate((indices, data['indices'][0]))
			stringData = np.concatenate((stringData, data['stringData']))

	return seqs, labels, indices, stringData

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

track_input = Input(shape=(1000,1))
G = Conv1D(320, 20, padding='valid', input_shape=(1000,1), activation='relu')(track_input)
G = MaxPooling1D(pool_size=10, strides=10)(G)
G = Dropout(0.2)(G)
G = Flatten()(G)
G = Dense(units=100, activation='sigmoid')(G)

fc_input = concatenate([H,G])

F = Dense(input_shape=(200,), units = 64, activation='relu')(fc_input)
F = Dropout(0.2)(F)
F = Dense(input_shape=(64,), units = 6, activation='sigmoid')(F)

model = Model(inputs=[seq_input, track_input], outputs=[F])
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

model.summary()

hist_data = pkl.load(open('F:\\data\\H3K36me3\\GM12878_H3K36me3_tracks.pkl', 'rb'))

seqs, labels, indices, stringData = loadmats([
		'F:\\data\\subc_sequences\\subc_seqs_no19.fa.data.1.mat',
		'F:\\data\\subc_sequences\\subc_seqs_no19.fa.data.2.mat',
	])

res = hist_data['res']
mean_tracks = hist_data['mean']

n_perm = 25

replabels = []

for i in range(labels.shape[0]):
	for j in range(n_perm):
		replabels.append(labels[i,:])

replabels = np.asarray(replabels)

seqs1k = []

for i in range(seqs.shape[0]):
	for j in range(n_perm):
		randindex = np.random.randint(0,99000)
		seqs1k.append(seqs[i,randindex:randindex+1000,:])

seqs1k = np.asarray(seqs1k)

seg_length = int(100000/res)

del hist_data

training_tracks = []

for i, str in enumerate(stringData):
	chrm = str.split()[0]
	start = int(int(str.split()[1]) / res)
	end = int(int(str.split()[2]) / res)
	track = mean_tracks[chrm]['means'][start + int(indices[i] / 100000) * seg_length:start + (int(indices[i] / 100000) + 1) * seg_length]

	for j in range(n_perm):
		training_tracks.append(track)

training_tracks = np.asarray(training_tracks)
training_tracks = np.expand_dims(training_tracks, axis=2)
print(training_tracks.shape, seqs1k.shape, replabels.shape)

del seqs, indices, labels, stringData

model.fit(x=[seqs1k, training_tracks], y=replabels, epochs=10, batch_size=256, validation_split=0.2, shuffle=True)
model.save('20170626_multi_train_model.h5')