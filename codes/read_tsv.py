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

yesfile = open(os.path.join(basepath, "ayush_dataset", "allyesfileindex.txt"), "w")
nofile = open(os.path.join(basepath, "ayush_dataset", "allnofileindex.txt"), "w")
exnofile = open(os.path.join(basepath, "ayush_dataset", "allexcludednofileindex.txt"), "w")



for line in f:
	cols = line.split("\t")[:-2]
	d = dict(zip(colnames, cols))
	if(d['Checked'].lower()[0] == 'y'):
		if(d['Marked']==""):
			left += 1
		else:
			if(d['Marked'] == "Y"):
				print(d['DebateId'], d['ID'], d['Id-1'], file=yesfile)
			elif(d['Marked'] == "N"):
				print(d['DebateId'], d['ID'], d['Id-1'], file=nofile)
			else:
				print(d)
			done += 1
	else:
		print(d['DebateId'], d['ID'], d['Id-1'], file=nofile)
		print(d['DebateId'], d['ID'], d['Id-1'], file=exnofile)


	#print(json.dumps(d))

print(done, left)
