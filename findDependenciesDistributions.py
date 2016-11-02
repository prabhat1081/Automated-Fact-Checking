import pickle
from corenlp import StanfordCoreNLP
directory = 'ayush_dataset/'
with open(directory+'newCfile.txt','r') as checkedSentences:
    checkedSentences = checkedSentences.readlines()
with open(directory+'newNCfile.txt','r') as unCheckedSentences:
    unCheckedSentences = unCheckedSentences.readlines()
nlp = StanfordCoreNLP('http://10.5.18.109:11111')

numChecked = len(checkedSentences)
numUnChecked = len(unCheckedSentences)

dictOfCheckedDeps = {}
for sentence in checkedSentences:
    text = ( sentence )
    output = nlp.annotate(
        text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse',
        'outputFormat': 'json'
    })
    for gloss in output['sentences'][0]['collapsed-dependencies']:
        try:
            dictOfCheckedDeps[gloss['dep']] += 1
        except:
            dictOfCheckedDeps[gloss['dep']] = 1
for dep in dictOfCheckedDeps:
    dictOfCheckedDeps[dep] = float(dictOfCheckedDeps[dep])/numChecked
    # print(dep, dictOfCheckedDeps)
pickle.dump(dictOfCheckedDeps,open('newDictOfCheckedDepsPython2.p','wb'),protocol=2)
print('Done for checked sentences')

dictOfUnCheckedDeps = {}
for sentence in unCheckedSentences:
    text = ( sentence )
    output = nlp.annotate(
        text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse',
        'outputFormat': 'json'
    })
    try:
        for gloss in output['sentences'][0]['collapsed-dependencies']:
            try:
                dictOfUnCheckedDeps[gloss['dep']] += 1
            except:
                dictOfUnCheckedDeps[gloss['dep']] = 1
    except:
        print(output)
for dep in dictOfUnCheckedDeps:
    dictOfUnCheckedDeps[dep] = float(dictOfUnCheckedDeps[dep])/numUnChecked
    # print(dep, dictOfCheckedDeps[dep])
pickle.dump(dictOfUnCheckedDeps,open('newDictOfUnCheckedDepsPython2.p','wb'),protocol=2)