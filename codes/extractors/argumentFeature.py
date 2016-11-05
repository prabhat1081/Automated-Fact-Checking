import os
import pandas as pd
arguingWords = ['no', 'not', 'never', 'contrary', 'objection', 'converse', 'reverse', 'refuse', 'disapprove', 'decline', 'reject', 'wrong', 'incorrect', 'error', 'misguide', 'mistake', 'untrue', 'unfair', 'absurd', 'foolish', 'illogical', 'irrational', 'nonsense', 'stupid', 'silly', 'unreasonable', 'idiotic', 'inane', 'insane']
index = None
dataset = None
basepath = "/home/bt1/13CS10060/btp"

def isArguing(block):
	if len(block)<=2:
		for words in block.iloc[0]['Sentence'].split(' '):
			if words in arguingWords:
				return True
	return False

def initIndexAndDataset(index_,dataset_):
	global index
	global dataset
	index = index_
	dataset = dataset_

def initIndexAndDataset():
	global dataset
	global index
	filename = os.path.join(basepath, "ayush_dataset", "annotated_single_all.tsv")
	dataset =  pd.read_csv(filename, sep="\t")
	filename = os.path.join(basepath, "ayush_dataset", "index.txt")
	indexStr = open(filename,'r').readlines()
	index = [[int(x) for x in line.strip().split(' ')] for line in indexStr]
	return index,dataset


def featureVal(indexId):
	did,bid,sid = index[indexId]
	itsDebate = dataset[dataset.DebateId == did]
	itsBlock = itsDebate[itsDebate.ID == bid]
	if sid == len(itsBlock)-1:
		nextBlock = itsDebate[itsDebate.ID == bid+1]
		if len(nextBlock) > 0:
			if nextBlock.iloc[0]['Speaker'] != itsBlock.iloc[0]['Speaker']:
				return isArguing(nextBlock)
	return False

def test(dataset_utils):
	initIndexAndDataset()
	for indexId in range(len(index)):
		t = featureVal(indexId)
		if t:
			print(dataset_utils.get_sentence(index[indexId]))
			print(t)