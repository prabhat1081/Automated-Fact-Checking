arguingWords = ['no', 'not', 'never', 'contrary', 'objection', 'converse', 'reverse', 'refuse', 'disapprove', 'decline', 'reject', 'wrong', 'incorrect', 'error', 'misguide', 'mistake', 'untrue', 'unfair', 'absurd', 'foolish', 'illogical', 'irrational', 'nonsense', 'stupid', 'silly', 'unreasonable', 'idiotic', 'inane', 'insane']
index = None
dataset = None


def isArguing(block):
	if len(block)<=2:
		for words in block[0].split(' '):
			if words in arguingWords:
				return True
	return False

def initIndexAndDataset(index_,dataset_):
	global index
	global dataset
	index = index_
	dataset = dataset_


def featureVal(sentence,indexId):
	did,bid,sid = index[indexId]
	itsDebate = dataset[dataset.DebateId == did]
	itsBlock = dataset[itsDebate.ID == bid]


