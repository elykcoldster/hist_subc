import sys

if __name__ == "__main__":
	input_bed = open(sys.argv[1], 'r')
	output_bed = open(sys.argv[2], 'w')
	flank_size = int(sys.argv[3])

	for line in input_bed:
		linedata = line.split()
		chr_name = linedata[0]
		start = int(linedata[1])
		end = int(linedata[2])
		other = linedata[3:]

		newline = chr_name + '\t' + str(int((start + end - flank_size) / 2)) + '\t' + str(int((start + end + flank_size) / 2))
		for data in other:
			newline += '\t' + data
		output_bed.write(newline + '\n')