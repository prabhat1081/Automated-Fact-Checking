from gensim.models import Word2Vec
import extractors.tokenizer as tokenizer
#import tokenizer
import numpy as np


model = Word2Vec.load_word2vec_format('embeddings/google_news_300.bin', binary=True)

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

print(features("Word is a game"))

