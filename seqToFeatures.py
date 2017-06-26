import os
import sys
import numpy as np
import h5py
import scipy.io

np.random.seed(1337) # for reproducibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, load_model
from keras.utils import to_categorical

def seq2bin(seq):
    binarray = np.empty([len(seq), 4], dtype=bool)
    for i, c in enumerate(seq):
        if c.lower() == 'a':
            binarray[i,:]= np.array([True,False,False,False])
        elif c.lower() == 'g':
            binarray[i,:]= np.array([False,True,False,False])
        elif c.lower() == 'c':
            binarray[i,:]= np.array([False,False,True,False])
        elif c.lower() == 't':
            binarray[i,:]= np.array([False,False,False,True])
        else:
            binarray = None
            break
    return binarray

def strToSubc(str):
	if str == 'NA':
		return 0
	elif str == 'A1':
		return 1
	elif str == 'A2':
		return 2
	elif str == 'B1':
		return 3
	elif str == 'B2':
		return 4
	elif str == 'B3':
		return 5

fa_file = open(sys.argv[1], 'r')
bed_file = open(sys.argv[2], 'r')
seq_lengths = int(sys.argv[3])
segment_length = int(sys.argv[4])

subcs = []
subclabels = []
bed_lines = []

for line in bed_file:
	subc = strToSubc(line.split()[3])
	subclabels.append(line.split()[3])
	bed_lines.append(line)
	if type(subc) == int:
		subcs.append(subc)

subcs = to_categorical(subcs)
print(subcs)

print ('Compiling Sequences')

line_num = 0
num_seqs = 10000

seqs = []
fullseqs = []

labels = []
segdata = []
indices = []
lineNumbers = []

num_files = 1

for line in fa_file:
	if line_num % 2 == 1:
		current_label = int(line_num / 2)
		print('Iteration {0}: {1} Sequences'. format(current_label, len(fullseqs)), end='\r')

		strlen = len(line) - 1
		
		for i in range(0, strlen, seq_lengths):
			subline = line[i:i+seq_lengths]
			subseq = seq2bin(subline)

			if subseq is not None:

				fullseqs.append(subseq)
				labels.append(subcs[current_label,:])
				segdata.append(bed_lines[current_label])
				indices.append(i)
				lineNumbers.append(current_label)

				"""for j in range(0, len(subseq), segment_length):
					segment = subline[j:j + segment_length]
					seqs.append(seq2bin(segment))"""

		# Perform memory flush to disk every 500 iterations
		if current_label > 0 and current_label % 500 == 0:
			print('Saving to file...')
			fullseqs = np.asarray(fullseqs)
			labels = np.asarray(labels)
			indices = np.asarray(indices)
			lineNumbers = np.asarray(lineNumbers)

			scipy.io.savemat(sys.argv[1] + '.data.' + str(num_files) + '.mat',
				{'bed_file': sys.argv[2],
				'seqs': fullseqs,
				'labels': labels,
				'indices': indices,
				'lineNumbers': lineNumbers
				'stringData': segdata})

			fullseqs = []
			labels = []
			segdata = []
			indices = []
			lineNumbers = []

			num_files += 1

	"""if len(seqs) >= num_seqs * (seq_lengths / segment_length):
		break"""
	line_num += 1

seqs = np.asarray(seqs)
fullseqs = np.asarray(fullseqs)
labels = np.asarray(labels)
indices = np.asarray(indices)

scipy.io.savemat(sys.argv[1] + '.data.' + str(num_files) + '.mat', {'seqs': fullseqs, 'labels': labels, 'indices': indices, 'stringData': segdata})

"""predictions = model.predict(seqs)
scipy.io.savemat(model_file + '.predictions.mat', {'pred': predictions})
print(predictions.shape)"""