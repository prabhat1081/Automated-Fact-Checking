import extractors.tokenizer as tokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer



names = ["compound", "neg", "neu", "pos"]


def feature_names():
	return names

def feature_name_type():
	return [(n,'REAL') for n in names]



def features(text):
	from nltk import tokenize
	sentences = tokenize.sent_tokenize(text)
	sid = SentimentIntensityAnalyzer()
	feature = [0] * len(names)
	for sentence in sentences:
		ss = sid.polarity_scores(sentence)
		for i, k in enumerate(sorted(ss)):
			feature[i] += ss[k]
	for i in range(len(feature)):
		feature[i] /= len(sentences)
	return feature



print("Testing Sentiment")
print(features("It is a cold day. I hate him"))