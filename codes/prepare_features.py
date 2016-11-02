import os
from extractors import *


basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"
workingdir = os.path.join(basepath, "new_data_results")


def get_stopwords(filename):
    words = open(basepath + "/ayush_dataset/"+filename, "r")
    return [ word.strip() for word in words.readlines()]

extractors = [bigrams.bigram_feature,
 			dependencies.dependenciesVector,
 			postags.pos_features,
 			subjective.subjective_feature,
 			]


def init():

	## Intiliaze bigrams
	positivefile = os.path.join(datapath, "yesfile.txt")
	
	threshold = 11
	bigrams.load_bigrams(workingdir, threshold, positivefile)

	## Initilaize subjective lexicon
	subjective_lexicon = os.path.join(datapath, "subclue.tff")
	subjective.load_lexicon(subjective_lexicon)

	## Initialization complete	


def write_feature_names(output):
	for extractor in extractors.__all__:
		print(extractors.extractor.feature_names())

def generate_features(filename, classid, output):
	#topicsall = topicLDA.batch_LDATopicVector(filename)
	topicsall = topicLDA.batch_LDATopicVector(filename)
	f = open(filename, "r")
	index = 0
	for line in f:
		features = []
		for extractor in extractors:
			features.extend(extractor(line))
		features.extend(topicsall[index])
		index = index + 1
		print(index)
		features.append(classid)
		print(" ".join(map(str, features)), file=output)
init()
positivefile = os.path.join(datapath, "yesfile.txt")
negfile = os.path.join(datapath, "nofile.txt")
#outputname = os.path.join(workingdir, "features.f")
#output = open(outputname, "w")
write_feature_names()
#generate_features(positivefile, 1, output)
#generate_features(negfile, 0, output)






