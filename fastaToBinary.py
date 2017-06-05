import sys
import numpy as np
from scipy.io import savemat, loadmat

def seq2bin(seq):
    binarray = []
    for c in seq:
        if c.lower() == 'a':
            binarray.append(np.array([True,False,False,False]))
        elif c.lower() == 'g':
            binarray.append(np.array([False,True,False,False]))
        elif c.lower() == 'c':
            binarray.append(np.array([False,False,True,False]))
        elif c.lower() == 't':
            binarray.append(np.array([False,False,False,True]))
        else:
            continue
    binarray = np.asarray(binarray)
    return binarray

if __name__ == "__main__":
    input_fasta = open(sys.argv[1], 'r')
    output_bin = sys.argv[2]

    max_seqs = None
    if len(sys.argv) > 3:
        max_seqs = int(sys.argv[3])

    all_binseq = []
    all_counts = []

    counts = loadmat('GM12878_feature_counts.mat')['feat_counts']

    n = 0
    for i, line in enumerate(input_fasta):
        if line[0] != '>':
            if n % 10000 == 0:
                print(n)
            seq = line
            binseq = seq2bin(seq)
            if binseq.shape[0] == 1000:
                all_binseq.append(binseq)
                all_counts.append(counts[int(i/2),:])
            n += 1

            if max_seqs is not None and n >= max_seqs:
                break
    all_binseq = np.asarray(all_binseq)
    savemat(output_bin, {'seqs' : all_binseq, 'counts': all_counts})
    print(all_binseq.shape)