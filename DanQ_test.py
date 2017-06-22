import sys
import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.wrappers import Bidirectional

input_file = sys.argv[1]
output_file = sys.argv[2]

forward_lstm = LSTM(320, input_shape=(None,320), return_sequences=True)
backward_lstm = LSTM(320, input_shape=(None,320), return_sequences=True)

print ('building model')

model = Sequential()
model.add(Conv1D(input_shape=(1000,4),
                padding="valid",
				strides=1,
				activation="relu",
				kernel_size=26,
				filters=320))

model.add(MaxPooling1D(pool_size=13, strides=13))

model.add(Dropout(0.2))

model.add(Bidirectional(forward_lstm))
model.add(Bidirectional(backward_lstm))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=75*640, units=925))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, units=91))
model.add(Activation('sigmoid'))

print ('compiling model')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.load_weights('C:\\Data\\GM12878_weights.h5')

testmat = scipy.io.loadmat(input_file)['seqs']
y = model.predict(testmat, verbose = 1)

f = h5py.File(output_file, "w")
f.create_dataset("pred", data=y)
f.close()