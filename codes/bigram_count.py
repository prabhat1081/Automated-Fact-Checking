import os
import json

from corenlp import StanfordCoreNLP

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

import pprint
from nltk.stem.porter import *
from getDependenciesVector import getDependenciesVector
from getLDATopicVector import getLDATopicVector


pp = pprint.PrettyPrinter(indent=4)

basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"


nlp = StanfordCoreNLP('http://10.5.18.109:11111')
stemmer = PorterStemmer()
output = ""



def get_stopwords(filename):
    words = open(basepath + "/ayush_dataset/"+filename, "r")
    return [ word.strip() for word in words.readlines()]

stopwords = []#get_stopwords("stopwords.txt")
puncts = list("!';\",.?()`$")
print(puncts)
stopwords.extend(puncts)


def sentence_tokenizer():
	# Train a Punkt tokenizer if required
	return nltk.data.load('tokenizers/punkt/english.pickle')
def word_tokenizer():
	return nltk.tokenize.treebank.TreebankWordTokenizer()


def getTokens(text):
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,lemma',
        'outputFormat': 'json'
    })
    return output['sentences']




def count_bigrams(filename):

    source = open(basepath+"/"+ filename, "r")
    cnt = Counter()
    for line in source.readlines():
        
        for sentence in getTokens(line):
            tokens = sentence['tokens']
            last = tokens[0]['word'].lower()
            for tokeninfo in tokens[1:]:
                if(tokeninfo['word'].lower() in stopwords): continue
                bigram = (last, tokeninfo['lemma'].lower())
                cnt[bigram] += 1
                last = tokeninfo['lemma'].lower()
            #cnt[(last, "</s>")] += 1
    return cnt



def plot(counts):
    labels, values = zip(*counts.items())

    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    #plt.xticks(indexes + width * 0.5, labels)
    plt.show()

def get_bigrams():
    nc = count_bigrams("ayush_dataset/newNCfile.txt")
    c = count_bigrams("ayush_dataset/newCfile.txt")
    #tot = c + nc

    #plot(tot)
    #plot(c)
    #plot(nc)


    num = 10
    #cnt = nc - (nc & c)
    bigrams = {}
    for bigram,cn in c.most_common():

        #if(bigram[0] == "<s>" or bigram[1] == "</s>"): continue

        if(cn < num): continue
        bigrams[bigram] =  cn
    return bigrams

def load_bigrams():
    import pickle
    
    try:
        with open(datapath + '/bigrams.pkl', 'rb') as f1:
            bigrams = pickle.load(f1)
    except:
        bigrams = get_bigrams()
        with open(datapath + '/bigrams.pkl', 'wb') as f1:
            pickle.dump(bigrams, f1)
    print(len(bigrams))
    pp.pprint(bigrams)
    return bigrams


def get_bigram_for_sentence(tokens_, bigrams):
    
    feature = [0]*len(bigrams)
    for sentence in tokens_:
        tokens = sentence['tokens']
        last = tokens[0]['word'].lower()
        for tokeninfo in tokens[1:]:
            if(tokeninfo['word'].lower() in stopwords): continue
            bigram = (last, tokeninfo['lemma'].lower())
            if(bigram in bigrams):
                feature[bigrams[bigram]] = 1
                #print("here", bigram, bigrams[bigram], feature[bigrams[bigram]])
            last = tokeninfo['lemma'].lower()

    return feature


def get_POS_Tag_features(tokens, pos_mapping):

    featurev = [0]*len(pos_mapping)
    for sentence in tokens:
        toks = sentence['tokens']
        for tokeninfo in toks:
            postag = tokeninfo['pos']
            if(postag in ["-LRB-", "-RRB-"]): continue
            featurev[pos_mapping[postag]] += 1
    return featurev


def load_subjectivity_lexicon():
    fi = open(datapath+"/subclue.tff")
    lexicon = {}
    for line in fi:
        pros = {}
        props = line.strip().split(" ")
        pros['type'] = props[0].split("=")[-1]
        pros['word'] = props[2].split("=")[-1]
        pros['pos'] = props[3].split("=")[-1]
        pros['stem'] = props[4].split("=")[-1]
        if(pros['word'] in lexicon):
            lexicon[pros['word']].append(pros)
        else:
            lexicon[pros['word']] = [pros]
    return lexicon


def pos_is_equal(tag, pos):
    if(pos == "verb"):
        return tag[0:2] == "VB"
    if(pos == "noun"):
        return tag[0:2] == "NN"
    if(pos == "adj"):
        return tag[0:2] == "JJ"
    if(pos == "adverb"):
        return tag[0:2] == "RB"
    if(pos == "anypos"):
        return True
    print(pos)
    return True


def get_subjective_count(tokens, lexicon):
    cnt_strong = 0
    cnt_weak = 0
    for sentence in tokens:
        toks = sentence['tokens']
        for tokeninfo in toks:
            word = tokeninfo['word']
            postag = tokeninfo['pos']
            stemmed = stemmer.stem(word)
            if(word in lexicon):
                valids = lexicon[word]
                for valid in valids:
                    if(pos_is_equal(postag, valid["pos"])):
                        if(valid['type'] == "strongsubj"):
                            cnt_strong = cnt_strong + 1
                        else:
                            cnt_weak =  cnt_weak + 1
                        break
            if(stemmed in lexicon):
                valids = lexicon[stemmed]
                for valid in valids:
                    if(valid['stem'] == "y" and pos_is_equal(postag, valid["pos"])):
                        if(valid['type'] == "strongsubj"):
                            cnt_strong = cnt_strong + 1
                        else:
                            cnt_weak =  cnt_weak + 1
                        break
    return [cnt_weak, cnt_strong]




def getTokens_(line):
    return [tokeninfo['word'] for sentence in line for tokeninfo in sentence['tokens']]




def get_features(filename, class_id, output):

    source = open(datapath+"/"+ filename, "r")
    bigrams = load_bigrams()
    l =len(bigrams)
    bigram_words = bigrams.keys()
    mapping = {}
    k = 0
    for word in bigram_words:
        mapping[word] = k
        k = k+1
    print(mapping)

    tagset = ["$", "``", "''", "(", ")", ",", "--", ".", ":", "CC", "CD" , "DT", "EX", "FW",
            "IN", "JJ", "JJR", "JJS", "LS","MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS",
            "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
            "VBN","VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]

    tag_mapping = {}
    k = 0
    for word in tagset:
        tag_mapping[word] = k
        k = k+1

    sub_lexicon = load_subjectivity_lexicon()
    
    for line in source.readlines():
        corenlpprocessed = getTokens(line.strip())
        f_bigram = get_bigram_for_sentence(corenlpprocessed, mapping)
        f_postag = get_POS_Tag_features(corenlpprocessed, tag_mapping)
        f_subjectivity = get_subjective_count(corenlpprocessed, sub_lexicon)
        f_dependencies = getDependenciesVector(line.strip())
        f_topic = getLDATopicVector(" ".join(getTokens_(corenlpprocessed)))
        f = f_bigram
        f.extend(f_postag)
        f.extend(f_subjectivity)
        f.extend(f_dependencies)
        f.extend(f_topic)
        f.append(class_id)
        #print(f)
        strd = " ".join(map(str, f))
        print(strd, file=output)
        
    

output = open(datapath+"/features_all.f", "w")    
get_features("newCfile.txt", 1, output)   
get_features("newNCfile.txt", 0, output)
output.close()  


