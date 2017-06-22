import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle

chromosome_lengths = {
		'chr1' : 249250621,
		'chr2' : 243199373,
		'chr3' : 198022430,
		'chr4' : 191154276,
		'chr5' : 180915260,
		'chr6' : 171115067,
		'chr7' : 159138663,
		'chrX' : 155270560,
		'chr8' : 146364022,
		'chr9' : 141213431,
		'chr10' : 135534747,
		'chr11' : 135006516,
		'chr12' : 133851895,
		'chr13' : 115169878,
		'chr14' : 107349540,
		'chr15' : 102531392,
		'chr16' : 90354753,
		'chr17' : 81195210,
		'chr18' : 78077248,
		'chr20' : 63025520,
		'chrY' : 59373566,
		'chr19' : 59128983,
		'chr22' : 51304566,
		'chr21' : 48129895
	}

if __name__ == "__main__":
	bedGraph_file = open(sys.argv[1], 'r')
	out_file = open(sys.argv[2],'wb')
	res = int(sys.argv[3])
	numlines = None
	if len(sys.argv) > 4:
		numlines = int(sys.argv[4])

	max_tracks = {}
	mean_tracks = {}
	line_num = 0
	for line in bedGraph_file:
		if numlines is not None and line_num >= numlines:
			break

		chrm = line.split()[0]
		if chrm == 'chrM' or chrm == 'chrUn' or chrm == 'chrMT':
			break
		if chrm not in max_tracks:
			print(chrm)
			max_tracks[chrm] = np.zeros(int(chromosome_lengths[chrm] / res) + 1)
			
			mean_tracks[chrm] = {}
			mean_tracks[chrm]['track'] = np.zeros(int(chromosome_lengths[chrm] / res) + 1)
			mean_tracks[chrm]['counts'] = np.zeros(int(chromosome_lengths[chrm] / res) + 1)
			mean_tracks[chrm]['means'] = np.zeros(int(chromosome_lengths[chrm] / res) + 1)

		signal = float(line.split()[3])
		start = int(line.split()[1])
		index = int(start / res)

		max_tracks[chrm][index] = max(max_tracks[chrm][index], signal)
		mean_tracks[chrm]['track'][index] += signal
		mean_tracks[chrm]['counts'][index] += 1
		mean_tracks[chrm]['means'][index] = mean_tracks[chrm]['track'][index] / mean_tracks[chrm]['counts'][index]

		line_num += 1

	tracks = {'max' : max_tracks, 'mean' : mean_tracks, 'res' : res}
	pickle.dump(tracks, out_file)