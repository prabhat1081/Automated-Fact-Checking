import os
import json
import itertools

import pandas as pd

basepath = "/home/bt1/13CS10060/btp"

filename = os.path.join(basepath, "ayush_dataset", "annotated_single_all.tsv")



f = open(filename, "r")

colnames = f.readline().split("\t")[:-2]

print(colnames)

done = 0
left = 0

yesfile = open(os.path.join(basepath, "ayush_dataset", "allyesfile.txt"), "w")
nofile = open(os.path.join(basepath, "ayush_dataset", "allnofile.txt"), "w")
exnofile = open(os.path.join(basepath, "ayush_dataset", "allexcludednofile.txt"), "w")



for line in f:
	cols = line.split("\t")[:-2]
	d = dict(zip(colnames, cols))
	if(d['Checked'].lower()[0] == 'y'):
		if(d['Marked']==""):
			left += 1
		else:
			if(d['Marked'] == "Y"):
				print(d['Sentence'], file=yesfile)
			elif(d['Marked'] == "N"):
				print(d['Sentence'], file=nofile)
			else:
				print(d)
			done += 1
	else:
		print(d['Sentence'], file=nofile)
		print(d['Sentence'], file=exnofile)


	#print(json.dumps(d))

print(done, left)
