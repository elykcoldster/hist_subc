import sys
import os
import numpy as np

from scipy.io import savemat

def group_consecutives(chrm, step=1):
	breakpoints = [0]
	current = 0
	for i, p in enumerate(chrm):
		if p != current:
			current = 1 - current
			breakpoints.append(i)
	return breakpoints

if __name__ == "__main__":
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
	
	res = 1000

	feature_dir = sys.argv[1]

	features = {}

	num_feats = len(os.listdir(feature_dir))

	chromosome_occupy = {}

	for key in chromosome_lengths:
		chromosome_occupy[key] = np.zeros((int(chromosome_lengths[key] / res + 1), num_feats))
		chromosome_occupy[key] = chromosome_occupy[key].astype(bool)

	for n in chromosome_lengths.items():
		features[n[0]] = {}

	print(features)

	for num, feat in enumerate(os.listdir(feature_dir)):
		print(num, feat)

		file = open(os.path.join(feature_dir,feat))

		for line in file:
			chrm = line.split()[0]
			start = int(line.split()[1])
			end = int(line.split()[2])

			rstart = int(start / 200) * 200
			rend = int(end / 200 + 1) * 200

			if rstart + 600 > chromosome_lengths[chrm]:
				continue

			chromosome_occupy[chrm][int(rstart / res),num] = True
			chromosome_occupy[chrm][int(rend / res),num] = True

			for segment in range(rstart, rend, 200):
				if str(segment) not in features[chrm]:
					features[chrm][str(segment)] = np.zeros(num_feats)
				features[chrm][str(segment)][num] += 1

	savemat('GM12878_occupy_1kb.mat', chromosome_occupy)

	feature_counts = []
	outbed = open('GM12878_peaks_91_feats.bed','w')
	for chrm in features:
		for segment in features[chrm]:
			end = str(int(segment) + 200)
			outbed.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(
				chrm,
				segment,
				end,
				'.',
				'0',
				'.',
				'0'
			))
			feature_counts.append(features[chrm][segment])

	feature_counts = np.asarray(feature_counts)
	savemat('GM12878_feature_counts.mat',{'feat_counts' : feature_counts})