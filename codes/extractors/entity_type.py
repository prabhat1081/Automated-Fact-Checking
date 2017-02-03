import extractors.tokenizer as tokenizer

#import tokenizer
from collections import Counter

names = ["PERSON", "LOCATION", "ORGANIZATION", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "DATE", "TIME", "DURATION", "SET"]


def feature_names():
    return ["entype_"+ i for i in names]

def feature_name_type():
    return [("entype_"+ i, 'NUMERIC') for i in names]



def features(text):
	parsed = tokenizer.ner(text)
	c = Counter()
	ners = []
	for sentence in parsed:
		tokens = sentence['tokens']

		for tokeninfo in tokens:
			word = tokeninfo['word']
			# if(tokeninfo['ner'] == 'O'): continue
			# c[tokeninfo['ner']] += 1
			ners.append(tokeninfo['ner'])
		from itertools import groupby
		print(ners)
		ner = [x[0] for x in groupby(ners)]
		print(ner)
		c_dash = Counter(ner)
		c += c_dash

	feature = [0]*len(names)
	print(c)
	for i,name in enumerate(names):
		feature[i] = c[name]

	return feature


print("Testing: Entity_type")
test = "1.5 million jobs created during the worst economic time this country has had since the Great Depression while the rest of the country lost 400,000 jobs."
test2 = "They would repeal the Affordable Care Act, not improve it."

print(features(test))
print(features(test2))