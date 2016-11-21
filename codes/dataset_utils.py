import json
import pandas as pd
import os

import error_analysis.similarity_n_gram

dataset = None

basepath = "/home/bt1/13CS10060/btp"


def initialize_map():
	filename = os.path.join(basepath, "ayush_dataset", "annotated_single_all.tsv")
	return pd.read_csv(filename, sep="\t", header=0)



def get_sentence(index):
	global dataset
	if(dataset is None):
		dataset = initialize_map()

	did, bid, sid = index

	

	result = dataset[dataset.DebateId == did]
	result = result[result.ID == bid]
	result = result[result.Id_1 == sid]
	return result.iloc[0]['Sentence']
	#result = dataset.query('"DebateId"==@did and "ID"==@bid and "Id-1"==@sid')
	#return result['Sentence']



def gen_full_debates(basedir):
	try:
		os.makedirs(basedir)
	except:
		pass
	global dataset
	if(dataset is None):
		dataset = initialize_map()

	for i in range(1,17):
		sents = []
		debate = dataset[dataset.DebateId == i]
		#print(debate)
		for _,row in debate.iterrows():
			#print(row)
			print(row['ID'], row['Id_1'])
			sents.append(row['Sentence'])
		article = " ".join(sents)
		print(article, file=open(basedir+"/debate"+str(i), "w"))

def detect_mismatch(basedir):
	try:
		os.makedirs(basedir)
	except:
		pass
	global dataset
	if(dataset is None):
		dataset = initialize_map()


	busterfile = open(os.path.join(basepath, "ayush_dataset", "buster_scores","debate14"), "r")
	s = busterfile.readlines()

	for i in range(14,15):
		sents = []
		debate = dataset[dataset.DebateId == i]
		#print(debate)
		i = 0
		for _,row in debate.iterrows():
			#print(row)
			#print(row['ID'], row['Id_1'])
			if(row['Sentence'] != s[i].strip().split("\t")[-1].strip()):
				print(row['ID'], row['Id_1'])
				print(s[i].strip().split("\t")[-1])
				print(i)

				break
			else:
				i = i + 1


def get_buster_score(indexes):
	global dataset
	import pickle
	if(dataset is None):
		dataset = initialize_map()
	fname = os.path.join(basepath, "ayush_dataset","buster_scores", "debate")
	stored = os.path.join(basepath, "ayush_dataset","buster_scores", "scores.pkl")
	try:
		with open(stored,"rb") as f:
			scoremap = pickle.load(f)
	except:
		ids = [3,4,13,14]
		scoremap = {}
		for id in ids:
			fnamei = fname + str(id)
			busterfile = open(fnamei, "r")
			s = busterfile.readlines()
			debate = dataset[dataset.DebateId == id]
			#print(debate)
			i = 0
			for _,row in debate.iterrows():
				#print(row)
				#print(row['ID'], row['Id_1'])
				idx = (row['DebateId'], row['ID'], row['Id_1'])
				scoremap[idx] = s[i].strip().split("\t")[0].strip()
				i = i + 1
		with open(stored, "wb") as f:
			pickle.dump(scoremap, f)


	return [scoremap[(index[0], index[1], index[2])] for index in indexes]


def evaluate_buster():
	global dataset
	import pickle
	if(dataset is None):
		dataset = initialize_map()
	fname = os.path.join(basepath, "ayush_dataset","buster_scores", "debate")
	ids = [3,4,13,14]
	scoremap = {}
	scores = []
	label = []
	for id in ids:
		fnamei = fname + str(id)
		busterfile = open(fnamei, "r")
		s = busterfile.readlines()
		debate = dataset[dataset.DebateId == id]
		#print(debate)
		i = 0
		for _,row in debate.iterrows():
			#print(row)
			#print(row['ID'], row['Id_1'])
			scores.append(s[i].strip().split("\t")[0].strip())
			if(row['Marked'] == "Y"):
				label.append(1)
			else:
				label.append(0)
			idx = (row['DebateId'], row['ID'], row['Id_1'])
			scoremap[idx] = s[i].strip().split("\t")[0].strip()
			i = i + 1

	o = open(os.path.join(basepath, "ayush_dataset","buster_scores", "score_class.txt"), "w")
	for score, cclass in zip(scores, label):
		print(score, cclass, sep="\t", file=o)



evaluate_buster()










#gen_full_debates(os.path.join(basepath, "ayush_dataset", "debates"))
#detect_mismatch(os.path.join(basepath, "ayush_dataset", "debates"))

#print(get_sentence((11,2,0)))

