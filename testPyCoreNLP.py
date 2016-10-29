from corenlp import StanfordCoreNLP

if __name__ == '__main__':
    nlp = StanfordCoreNLP('http://10.5.18.109:11111')
    text = (
        'Yesterday, the president stunningly admitted this. He said, "we do not have leverage with China to get a better deal on Iran because we need them to lend us money to continue operating our government".')
        #'A million students registered for the examination. They are very good.')
       # 'Pusheen and Smitha walked along the beach. Pusheen wanted to surf,'
       # 'but fell off the surfboard.')
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,dcoref',
        'outputFormat': 'json',
        'coref.doClustering' : True
    })
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(output['sentences'][0]['tokens'])
    parsed_sentence = output['sentences'][0]['parse']
    parsed_tokens = parsed_sentence.split('\n')
    #for parsed_token in parsed_tokens:
        #print("Token level =", (len(parsed_token)-len(parsed_token.strip()))//2)
    # output = nlp.tokensregex(text, pattern='/Pusheen|Smitha/', filter=False)
    # print(output)
    # output = nlp.semgrex(text, pattern='{tag: VBD}', filter=False)
    # print(output)
