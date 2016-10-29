def get_subj_lexicon():
    with open('/home/bt1/13CS10060/btp/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff','r') as subj_lexicon:
        subj_lexicon = subj_lexicon.readlines()

    subjectiveWords = {}
    for line in subj_lexicon:
        line = line.strip()
        line = line.split(' ')
        newline = []
        for props in line:
            props = props.split('=')[-1]
            newline.append(props)
        print(newline)
        # type_,len_,word1_,pos1_,stemmed1_,priorpolarity_ = newline
        subjectiveWords[newline[2]] = (newline[0],newline[3])   # type and postag
    return subjectiveWords
##########################################################################################################################################

import pickle
from corenlp import StanfordCoreNLP
directory = '/home/bt1/13CS10060/btp/ayush_dataset/'
with open(directory+'c1file.txt','r') as checkedSentences:
    checkedSentences = checkedSentences.readlines()
with open(directory+'nc1file.txt','r') as unCheckedSentences:
    unCheckedSentences = unCheckedSentences.readlines()
nlp = StanfordCoreNLP('http://10.5.18.109:11111')

dictOfCheckedPairs = {}
for sentence in checkedSentences:
    # print(sentence)
    text = ( sentence )
    output = nlp.annotate(
        text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse',
        'outputFormat': 'json'
    })
    for gloss in output['sentences'][0]['collapsed-dependencies']:
        if gloss['dep'] == 'dobj':
            try:
                dictOfCheckedPairs[(gloss['dependentGloss'].lower(), gloss['governorGloss'].lower())] += 1
            except:
                dictOfCheckedPairs[(gloss['dependentGloss'].lower(), gloss['governorGloss'].lower())] = 1
pickle.dump(dictOfCheckedPairs,open('dictOfCheckedPairs.p','wb'))
print('Done for checked sentences')

dictOfUnCheckedPairs = {}
for sentence in unCheckedSentences:
    # print(sentence)
    text = ( sentence )
    output = nlp.annotate(
        text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse',
        'outputFormat': 'json'
    })
    try:
        for gloss in output['sentences'][0]['collapsed-dependencies']:
            if gloss['dep'] == 'dobj':
                try:
                    dictOfUnCheckedPairs[(gloss['dependentGloss'].lower(), gloss['governorGloss'].lower())] += 1
                except:
                    dictOfUnCheckedPairs[(gloss['dependentGloss'].lower(), gloss['governorGloss'].lower())] = 1
    except:
        print(output)

pickle.dump(dictOfUnCheckedPairs,open('dictOfUnCheckedPairs.p','wb'))

# if __name__ == '__main__':
#     nlp = StanfordCoreNLP('http://10.5.18.109:9000')
#     text = (
#         'Pusheen and Smitha walked along the beach. Pusheen wanted to surf,'
#         'but fell off the surfboard.')
#     output = nlp.annotate(text, properties={
#         'annotators': 'tokenize,ssplit,pos,depparse,parse',
#         'outputFormat': 'json'
#     })
#     print(output['sentences'][0]['parse'])
#     parsed_sentence = output['sentences'][0]['parse']
#     parsed_tokens = parsed_sentence.split('\n')
#     for parsed_token in parsed_tokens:
#         print("Token level =", (len(parsed_token)-len(parsed_token.strip()))//2)
#     # output = nlp.tokensregex(text, pattern='/Pusheen|Smitha/', filter=False)
#     # print(output)
#     # output = nlp.semgrex(text, pattern='{tag: VBD}', filter=False)
#     # print(output)