import extractors.tokenizer as tk
import pprint
import scipy
import numpy as np

pp = pprint.PrettyPrinter(indent=4)

import sys
PATH_TO_SKIPTHOUGHTS = "/home/bt1/13CS10060/btp/skip-thoughts"
sys.path.insert(0, PATH_TO_SKIPTHOUGHTS)

import skipthoughts



model = None


def encode(X, use_EOS=False):
	global model
	if(model is None):
		model = skipthoughts.load_model()


	vectors = skipthoughts.encode(model, X, verbose=False)
	return vectors


def calculate_avg_similarity_block(X):
	vs = encode(X)
	if(len(X) == 1):
		return [1]
	pair_cosine = 1 - scipy.spatial.distance.cdist(vs, vs, "cosine")
	summed = np.sum(pair_cosine, axis=1) - 1
	res = summed / (len(X) - 1)

	return res


def features(X):
	return calculate_avg_similarity_block(X)


def feature_name_type():
	return [('sent_sim', "REAL")]


X = ["I am going to china.", "He is going to Japan."]
v = encode(X)
print(scipy.spatial.distance.cosine(v[0],v[1]))

pp.pprint(v)