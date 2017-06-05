import sys
import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from sklearn.model_selection import train_test_split

from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.wrappers import Bidirectional

input_file = sys.argv[1]
output_file = sys.argv[2]

print ('loading data')
#trainmat = h5py.File('GM12878_train.mat')
trainmat = scipy.io.loadmat(input_file)
#validmat = scipy.io.loadmat('GM12878_valid.mat')
#testmat = scipy.io.loadmat('GM12878_test.mat')

X_train = np.array(trainmat['seqs'])
y_train = np.array(trainmat['counts'])

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
#earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

test_split = 0.4

print('Splitting data into training and test sets. Test split: {0}'.format(test_split))

Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, y_train, test_size = test_split, random_state = 1337)

model.fit(Xtrain, ytrain, batch_size=100, epochs=15, shuffle=True, verbose=1, validation_split = 0.1)

model.save(output_file)

tresults = model.evaluate(Xtest, ytest)

print (tresults)