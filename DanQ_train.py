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

print ('loading data')
trainmat = h5py.File('GM12878_train.mat')
validmat = scipy.io.loadmat('GM12878_valid.mat')
testmat = scipy.io.loadmat('GM12878_test.mat')

X_train = np.transpose(np.array(trainmat['GM12878_trainxdata']),axes=(2,0,1))
y_train = np.array(trainmat['GM12878_traindata']).T

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

print ('running at most 15 epochs')

#checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(X_train, y_train, batch_size=100, epochs=15, shuffle=True, verbose=1, validation_split=0.1)

model.save('GM12878_weights.h5')

tresults = model.evaluate(np.transpose(testmat['GM12878_testxdata'],axes=(0,2,1)), testmat['GM12878_testdata'])

print (tresults)
