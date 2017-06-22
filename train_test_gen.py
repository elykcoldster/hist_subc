import sys
import numpy as np

from scipy.io import loadmat

def subcToNum(subc):
	if subc == "NA":
		return 0
	elif subc == "A1":
		return 1
	elif subc == "A2":
		return 2
	elif subc == "B1":
		return 3
	elif subc == "B2":
		return 4
	elif subc == "B3":
		return 5

if __name__ == "__main__":
	train_bed_file = open(sys.argv[1], 'r')
	test_bed_file = open(sys.argv[2], 'r')
	occ_mat = loadmat(sys.argv[3])
	res = int(sys.argv[4])

	traindata = []
	trainlabels = []
	testdata = []
	testlabels = []

	for line in train_bed_file:
		chrm = line.split()[0]
		start = int(int(line.split()[1]) / res)
		end = int(int(line.split()[2]) / res)
		label = subcToNum(line.split()[3])

		for i in range(start, end):
			traindata.append(occ_mat[chrm][i,:])
			trainlabels.append(label)

	for line in test_bed_file:
		chrm = line.split()[0]
		start = int(int(line.split()[1]) / res)
		end = int(int(line.split()[2]) / res)
		label = subcToNum(line.split()[3])

		for i in range(start, end):
			testdata.append(occ_mat[chrm][i,:])
			testlabels.append(label)

	traindata = np.asarray(traindata)
	testdata = np.asarray(testdata)