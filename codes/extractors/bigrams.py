import os
import json
import pprint 
import numpy as np
import extractors.tokenizer as tokenizer
from collections import Counter

pp = pprint.PrettyPrinter(indent=4)


stopwords = ["-lrb-", "...", "ph"]#get_stopwords("stopwords.txt")
puncts = list("!';\",.?()`$")

stopwords.extend(puncts)
print(stopwords)




def count_bigrams(filename):

    source = open(filename, "r")
    cnt = Counter()
    for line in source:
        for sentence in tokenizer.parse(line.strip()):
            tokens = sentence['tokens']
            last = tokens[0]['lemma'].lower()

            for tokeninfo in tokens[1:]:
                if(tokeninfo['lemma'].lower() in stopwords): continue
                bigram = (last, tokeninfo['lemma'].lower())
                cnt[bigram] += 1
                last = tokeninfo['lemma'].lower()
            #cnt[(last, "</s>")] += 1
    return cnt

def generate_bigrams(threshold, filename):
    c = count_bigrams(filename)
    bigrams = {}
    k = 0
    for bigram,cn in c.most_common():

        if(cn < threshold): continue
        bigrams[bigram] =  k
        k = k + 1

    return bigrams

def load_bigrams(datapath, threshold, filename):
    import pickle
    global bigrams
    try:
        with open(datapath + '/bigrams.pkl', 'rb') as f1:
            bigrams = pickle.load(f1)
    except:
        bigrams = generate_bigrams(threshold, filename)
        with open(datapath + '/bigrams.pkl', 'wb') as f1:
            pickle.dump(bigrams, f1)
    print(len(bigrams))
    pp.pprint(bigrams)
    return bigrams


def bigram_feature(text):
    parsed = tokenizer.parse(text)
    feature = [0]*len(bigrams)
    for sentence in parsed:
        tokens = sentence['tokens']
        last = tokens[0]['word'].lower()
        for tokeninfo in tokens[1:]:
            if(tokeninfo['word'].lower() in stopwords): continue
            bigram = (last, tokeninfo['lemma'].lower())
            try:
                feature[bigrams[bigram]] += 1
            except:
                pass
            last = tokeninfo['lemma'].lower()

    return feature