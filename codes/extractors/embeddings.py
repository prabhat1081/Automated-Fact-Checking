from gensim.models import Word2Vec
import extractors.tokenizer as tokenizer
#import tokenizer
import numpy as np
import os


basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"

#model = Word2Vec.load_word2vec_format('embeddings/google_news_300.bin', binary=True)

NDIM = 300
def features(text):
	parsed = tokenizer.parse(text.strip())
	embeds = []
	for sentence in parsed:
		toks = sentence['tokens']
		for tokeninfo in toks:
			try:
				word = tokeninfo['word'].lower()
				embeds.append(model[word])
			except:
				try:
					embeds.append(model[tokeninfo['lemma']].lower())
				except:
					pass

	embeds = np.asarray(embeds)
	return np.mean(embeds, axis=0)

def feature_names():
	return ["embed_"+str(i) for i in range(NDIM)]

def get_stopwords(filename):
    words = open(filename, "r")
    return [ word.strip() for word in words.readlines()]

stopwords = get_stopwords(os.path.join(datapath, "stopwords.txt"))

def sentence_sim(sent1, sent2):
	parsed = tokenizer.parse(sent1.strip())
	sent1_ = [tokeninfo['word'] for sentence in parsed 
		for tokeninfo in sentence['tokens'] if tokeninfo['word'] not in stopwords]
	parsed = tokenizer.parse(sent2.strip())
	sent2_ = [tokeninfo['word'] for sentence in parsed 
		for tokeninfo in sentence['tokens'] if tokeninfo['word'] not in stopwords]

	return  model.wmdistance(sent1_, sent2_)



#print(features("Word is a game"))


