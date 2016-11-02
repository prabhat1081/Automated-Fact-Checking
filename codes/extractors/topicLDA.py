import subprocess
basepath = "/home/bt1/13CS10060/btp/"
LDA_MODEL_DIRECTORY = '/home/bt1/13CS10060/btp/LDAAfterStopWordRemoval/'
TEST_FILE_NAME = 'test_sentence.txt'

BATCH_TEST_FILE_NAME = 'batch_test_sentence.txt'

import extractors.tokenizer as tokenizer

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def process_text(text):
	parsed = tokenizer.parse(text)
	return " ".join([tokeninfo['word'] for sentence in parsed for tokeninfo in sentence['tokens']])

def LDATopicVector(text):
	sentence = process_text(text)
	sentence = sentence.strip()	#To remove any trailing '\n'
	with open(LDA_MODEL_DIRECTORY+TEST_FILE_NAME,'w') as f:
		f.write(str(len(sentence.split('\n')))+'\n'+sentence)
	subprocess.call(basepath + 'GibbsLDA++-0.2/src/lda -inf -dir '+LDA_MODEL_DIRECTORY+' -model model-final -niters 100 -dfile '+TEST_FILE_NAME,
		shell=True, stdout=DEVNULL)
	with open(LDA_MODEL_DIRECTORY+TEST_FILE_NAME+'.theta') as f:
		vector = f.read().strip()
	vector = [float(x) for x in vector.split(' ')]
	# print vector
	return vector

vector = LDATopicVector('1.5 million jobs created during the worst economic time this country has had since the Great Depression while the rest of the country lost 400,000 jobs.')
print(vector)


def batch_LDATopicVector(filename):
	f = open(filename, "r")
	tempfile = open(LDA_MODEL_DIRECTORY+BATCH_TEST_FILE_NAME, "w")
	lines = f.readlines()
	cnt = len(lines)
	print(cnt, file=tempfile)
	for line in lines:
		text = process_text(line)
		print(text, file=tempfile)
	tempfile.close()

	subprocess.call(basepath + 'GibbsLDA++-0.2/src/lda -inf -dir '+LDA_MODEL_DIRECTORY+' -model model-final -niters 100 -dfile '+BATCH_TEST_FILE_NAME,
		shell=True)

	topicfile = open(LDA_MODEL_DIRECTORY+BATCH_TEST_FILE_NAME+'.theta', "r")

	topicsall = []

	for line in topicfile:
		vector = map(float, line.strip().split(' '))
		topicsall.append(vector)

	return topicsall


def feature_names():
    return ["top_"+ str(i) for i in range(20)]






