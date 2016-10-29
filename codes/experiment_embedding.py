from gensim.models import Word2Vec
import nltk


import xmlrpc.client

#model = xmlrpc.client.ServerProxy('http://10.5.18.109:9997')
model = Word2Vec.load_word2vec_format('embeddings/google_news_300.bin', binary=True)  # C binary format

notvec = model.getvector('not')
yes= model.getvector('yes')

sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
sentence_president = 'The president greets the press in Chicago'.lower().split()

 # Compute WMD.

distance = model.wmdistance(sentence_obama, sentence_president)

print(distance)

from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')
sentence_obama = [w for w in sentence_obama if w not in stopwords]
sentence_president = [w for w in sentence_president if w not in stopwords]

distance = model.wmdistance(sentence_obama, sentence_president)

print(distance)

while(True):
	word = input()
	vec = model.getvector(word)
	nvec= vec + notvec - yes
	print(model.similar_by_vector(nvec))


