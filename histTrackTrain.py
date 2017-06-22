import sys
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Dense, Flatten, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical

def labelToInt(label):
	if label == 'NA':
		return 0
	elif label == 'A1':
		return 1
	elif label == 'A2':
		return 1
	elif label == 'B1':
		return 2
	elif label == 'B2':
		return 2
	elif label == 'B3':
		return 2
	elif label == 'B4':
		return 2

if __name__ == '__main__':
	print('Loading Data...')
	bed = open(sys.argv[1], 'r')
	
	all_tracks = []

	for i in range(2, len(sys.argv)):
		all_tracks.append(pickle.load(open(sys.argv[i], 'rb')))
	

	res = all_tracks[0]['res']

	max_tracks = []
	for tracks in all_tracks:
		max_tracks.append(tracks['max'])

	num_tracks = len(max_tracks)
	input_dim = int(100000 / res)

	print('Input shape:', (input_dim, num_tracks))

	training_tracks = []
	training_labels = []

	print('Building Training Data...')
	for line in bed:
		chrm = line.split()[0]
		start_index = int(int(line.split()[1]) / res)
		end_index = int(int(line.split()[2]) / res)
		label = labelToInt(line.split()[3])

		if chrm == 'chr19':
			continue

		for i in range(start_index, end_index, input_dim):
			if i + input_dim <= end_index:
				track_segs = []
				for track in max_tracks:
					track_seg = track[chrm][i:i+input_dim]
					track_segs.append(track_seg)
				track_segs = np.asarray(track_segs).transpose()
				training_tracks.append(track_segs)
				training_labels.append(label)
	training_tracks = np.asarray(training_tracks)
	training_labels = to_categorical(training_labels)
	
	print('Building Model...')
	model = Sequential()
	model.add(Conv1D(320, 32, activation='relu', input_shape=(input_dim,num_tracks)))
	model.add(MaxPooling1D(8,8))
	model.add(Dropout(0.25))

	model.add(Conv1D(160, 32, activation='relu', input_shape=(input_dim,num_tracks)))
	model.add(MaxPooling1D(8,8))
	model.add(Dropout(0.25))

	model.add(Flatten())

	model.add(Dense(3, activation='sigmoid'))

	print(model.summary())

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print('Training Model...')

	model.fit(training_tracks, training_labels, epochs=50, batch_size=32, verbose=1, validation_split=0.1)
	model.save('F:\\Data\\hist_subc_weights.h5')