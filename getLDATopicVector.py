import subprocess
basepath = ""
LDA_MODEL_DIRECTORY = 'LDAAfterStopWordRemoval/'
TEST_FILE_NAME = 'test_sentence.txt'

def getLDATopicVector(sentence):
	sentence = sentence.strip()	#To remove any trailing '\n'
	with open(LDA_MODEL_DIRECTORY+TEST_FILE_NAME,'w') as f:
		f.write(str(len(sentence.split('\n')))+'\n'+sentence)
	subprocess.call(basepath + 'GibbsLDA++-0.2/src/lda -inf -dir '+LDA_MODEL_DIRECTORY+' -model model-final -niters 100 -dfile '+TEST_FILE_NAME,shell=True)
	with open(LDA_MODEL_DIRECTORY+TEST_FILE_NAME+'.theta') as f:
		vector = f.read().strip()
	vector = [float(x) for x in vector.split(' ')]
	# print vector
	return vector

vector = getLDATopicVector('1.5 million jobs created during the worst economic time this country has had since the Great Depression while the rest of the country lost 400,000 jobs.')
print(vector)