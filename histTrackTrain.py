import sys
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Dense, Flatten, MaxPooling1D
from keras.utils import to_categorical

def labelToInt(label):
	if label == 'NA':
		return -1
	elif label == 'A1':
		return 0
	elif label == 'A2':
		return 1
	elif label == 'B1':
		return 2
	elif label == 'B2':
		return 3
	elif label == 'B3':
		return 4
	elif label == 'B4':
		return 5

if __name__ == '__main__':
	print('Loading Data...')
	bed = open(sys.argv[1], 'r')
	tracks = pickle.load(open(sys.argv[2], 'rb'))
	
	max_track = tracks['max']
	res = tracks['res']

	input_dim = int(100000 / res)

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
				track_seg = max_track[chrm][i:i+input_dim]
				track_seg = np.expand_dims(track_seg, axis=1)
				training_tracks.append(track_seg)
				training_labels.append(label)
	training_tracks = np.asarray(training_tracks)
	training_labels = to_categorical(training_labels)
	
	print('Building Model...')
	model = Sequential()
	model.add(Conv1D(320, 16, activation='relu', input_shape=(input_dim,1)))

	model.add(MaxPooling1D(4,4))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(5, activation='sigmoid'))

	print(model.summary())

	model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

	print('Training Model...')

	model.fit(training_tracks[0:20000,:], training_labels[0:20000], epochs=15, batch_size=32, verbose=1, validation_split=0.1)