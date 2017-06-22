import os
import sys
import numpy as np
import h5py
import scipy.io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(1337) # for reproducibility

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
# from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.wrappers import Bidirectional

print('Building and compiling model...')

input = Input(shape=(1000, 4))
H = Conv1D(320, 26, padding='valid', input_shape=(1000,4), activation='relu')(input)
H = MaxPooling1D(pool_size=13, strides=13)(H)
H = Dropout(0.2)(H)

#H = Bidirectional(LSTM(320, input_shape=(None, 320), return_sequences=True))(H)
#H = Bidirectional(LSTM(320, input_shape=(None, 320), return_sequences=True))(H)
#H = Dropout(0.5)(H)
H = Flatten()(H)

H = Dense(input_dim=75*320, units=925, activation='relu')(H)
H = Dense(input_dim=925, units=2, activation='sigmoid')(H)

model = Model(inputs=input, outputs=H)
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

model.summary()

pos_mat_file = sys.argv[1]
neg_mat_file = sys.argv[2]

print('Loading training set...')

pos_mat = scipy.io.loadmat(pos_mat_file)['seqs']
neg_mat = scipy.io.loadmat(neg_mat_file)['seqs']

print('Positive training set:', pos_mat.shape)
print('Negative training set:', neg_mat.shape)

pos_labels = np.zeros((pos_mat.shape[0], 2))
neg_labels = np.zeros((neg_mat.shape[0], 2))

pos_labels[:,0] = 1
neg_labels[:,1] = 1

train_data = np.concatenate((pos_mat, neg_mat))
print('Total training set:', train_data.shape)

train_labels = np.concatenate((pos_labels, neg_labels))

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2)

model.fit(X_train, y_train, epochs = 15, batch_size=256, shuffle=True, verbose=1)
model.save('GM12878_H3K36me3_model.h5')

score = model.evaluate(X_test, y_test, batch_size=100, verbose=1)
print(score)