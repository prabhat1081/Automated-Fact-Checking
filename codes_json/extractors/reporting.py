import json
import os

order_words = ['commanded', 'ordered', 'warned', 'asked', 'advised', 'invited', 'begged', 'forbade', 'prohibited']
promise_words = ['hope', 'propose', 'promise', 'threaten', 'guarantee', 'swear']

best_order_words = ['commanded', 'ordered', 'warned', 'advised', 'invited', 'forbade', 'prohibited', 'suggested']
best_promise_words = ['proposed', 'promised', 'threatened', 'guaranteed', 'sweared']
said_words = ['told', 'said']

basepath = '/home/bt1/13CS10060/btp'

def isOrder(sentence):
	sentence = sentence.lower().strip().split(' ')
	pos = -1
	i = 0
	for word in sentence:
		if word in best_order_words:
			pos = i
		i+=1
	if pos == -1:
		return False
	sentence = sentence[pos:]
	if 'to' in sentence:
		return True
	return False

def isPromise(sentence):
	sentence = sentence.lower().strip().split(' ')
	pos = -1
	i = 0
	for word in sentence:
		if word in best_promise_words+said_words:
			pos = i
		i+=1
	if pos == -1:
		return False
	sentence = sentence[pos:]
	if ('to' in sentence) or ('that' in sentence) or ('would' in sentence):
		return True
	return False

def check():
	order = {}
	promise = {}
	dtafilename = os.path.join(basepath, "ayush_dataset", "data.json")
	data = json.load(open(dtafilename))
	samples = data['samples']
	for sample in samples:
		sentence = sample['Sentence']
		if isOrder(sentence):
			try:
				order[sample['Marked']] += 1
			except:
				order[sample['Marked']] = 1
		if isPromise(sentence):
			if sample['Marked']=='Y':
				print(sample['Marked'] + '\t' + sentence)
			try:
				promise[sample['Marked']] += 1
			except:
				promise[sample['Marked']] = 1
	print(order)
	print(promise)

if __name__=='__main__':
	check()