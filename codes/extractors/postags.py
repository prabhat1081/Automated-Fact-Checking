import extractors.tokenizer as tokenizer


tagset = ["$", "``", "''", "(", ")", ",", "--", ".", ":", "CC", "CD" , "DT", "EX", "FW",
            "IN", "JJ", "JJR", "JJS", "LS","MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS",
            "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
            "VBN","VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]

tag_mapping = {}
k = 0
for word in tagset:
    tag_mapping[word] = k
    k = k+1

def pos_features(text):
    parsed = tokenizer.parse(text)
    featurev = [0]*len(tag_mapping)
    for sentence in parsed:
        toks = sentence['tokens']
        for tokeninfo in toks:
            postag = tokeninfo['pos']
            if(postag in ["-LRB-", "-RRB-"]): continue
            featurev[tag_mapping[postag]] += 1
    return featurev