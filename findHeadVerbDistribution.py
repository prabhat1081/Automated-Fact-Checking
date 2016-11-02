from corenlp import StanfordCoreNLP
directory = 'ayush_dataset/'
with open(directory+'newCfile.txt','r') as checkedSentences:
	checkedSentences = checkedSentences.readlines()
with open(directory+'newNCfile.txt','r') as unCheckedSentences:
	unCheckedSentences = unCheckedSentences.readlines()
nlp = StanfordCoreNLP('http://10.5.18.109:11111')

dictionaryOfCheckedVerbs = {}
for sentence in checkedSentences:
	# print(sentence)
	text = ( sentence )
	output = nlp.annotate(
		text, properties={
		'annotators': 'tokenize,ssplit,pos,depparse,parse',
		'outputFormat': 'json'
	})
	for token in output['sentences'][0]['tokens']:
		if token['pos'][:2] == 'VB':
			try:
				dictionaryOfCheckedVerbs[token['word']]+=1
			except:
				dictionaryOfCheckedVerbs[token['word']]=1
print('Checked Verbs :--')
for w in sorted(dictionaryOfCheckedVerbs, key=dictionaryOfCheckedVerbs.get, reverse=True):
	print(w, dictionaryOfCheckedVerbs[w])
# print(dictionaryOfCheckedVerbs)
		# print(token['word'], token['pos'])
	# for gloss in output['sentences'][0]['collapsed-dependencies']:
	# 	print(gloss)
	# break
	# for gloss in output['sentences'][0]['collapsed-dependencies']:
	# 	print(gloss['dep'][:2]) 
	# 	print
	# 	if gloss['dep'] == 'ROOT' or gloss['dep'][:2].upper() == 'VB':
	# 		print(gloss['dependentGloss'], gloss['dep']),
	# 	print
		# print (gloss['dependentGloss'], gloss['dep'])
	# print(output['sentences'][0])
	# break

dictionaryOfUnCheckedVerbs = {}
for sentence in unCheckedSentences:
	# print(sentence)
	text = ( sentence )
	output = nlp.annotate(
		text, properties={
		'annotators': 'tokenize,ssplit,pos,depparse,parse',
		'outputFormat': 'json'
	})
	# print(sentence)
	# print(output['sentences'][0])
	try:
		for token in output['sentences'][0]['tokens']:
			if token['pos'][:2] == 'VB':
				try:
					dictionaryOfUnCheckedVerbs[token['word']]+=1
				except:
					dictionaryOfUnCheckedVerbs[token['word']]=1
	except:
		print(sentence)
		print(output)
print()
print('Unchecked Verbs :--')
for w in sorted(dictionaryOfUnCheckedVerbs, key=dictionaryOfUnCheckedVerbs.get, reverse=True):
	print(w, dictionaryOfUnCheckedVerbs[w])
# print(dictionaryOfUnCheckedVerbs)
		# print(token['word'], token['pos'])
	# for gloss in output['sentences'][0]['collapsed-dependencies']:
	# 	print(gloss)
	# break
	# for gloss in output['sentences'][0]['collapsed-dependencies']:
	# 	print(gloss['dep'][:2]) 
	# 	print
	# 	if gloss['dep'] == 'ROOT' or gloss['dep'][:2].upper() == 'VB':
	# 		print(gloss['dependentGloss'], gloss['dep']),
	# 	print
		# print (gloss['dependentGloss'], gloss['dep'])
	# print(output['sentences'][0])
	# break