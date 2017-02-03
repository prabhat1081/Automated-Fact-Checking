import arff
import os
from extractors import *
import numpy
import read_tsv


basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"
workingdir = os.path.join(basepath, "current")

def get_stopwords(filename):
    words = open(basepath + "/ayush_dataset/"+filename, "r")
    return [ word.strip() for word in words.readlines()]



def init():

    ## Intiliaze bigrams
    positivefile = os.path.join(datapath, "yesfile.txt")
    
    threshold = 6
    bigrams.load_bigrams(workingdir, threshold, positivefile, os.path.join(datapath, "stopwords.txt"))

    ## Initilaize subjective lexicon
    subjective_lexicon = os.path.join(datapath, "subclue.tff")
    subjective.load_lexicon(subjective_lexicon)

    ## Initialization complete

extractors = [bigrams.bigram_feature,
          dependencies.dependenciesVector,
          postags.pos_features,
          subjective.subjective_feature
          ]

extractors_ = [bigrams, dependencies,embeddings, postags, sentiment, subjective, verb_cat]

def get_feature_names(extractors):
    names = []
    for extractor in extractors:
        ff = extractor.feature_name_type()
        print(extractor.__name__, len(ff))
        names.extend(ff)
    names.append(("class_label", ['0','1']))
    print(len(names))
    return names


def get_feature_val(extractors, text, class_):
    features = []
    for extractor in extractors:
        features.extend(extractor.features(text))
    features.append(class_)
    return features





def feature_topic():
    instances = read_tsv.get_instance()
    colnames = next(instances)
    colnames = [(c, 'STRING') for c in colnames]
    attributes = topicLDA.feature_name_type()
    attributes.append(("class_label", ['0','1']))
    colnames.extend(attributes)
    attributes = colnames

    count = 0
    
    for instance_ in instances:
        instance, text, class_ = instance_
        topicLDA.accumulate_batch(text)
        count += 1
        if(count % 100 == 0):
            print("Accumulated: ",count)
        

    print("Batch Accumulated")

    topicsall = topicLDA.run_batch()

    instances = read_tsv.get_instance()
    next(instances)
    count = 0
    data_feat = []
    # print(len(topicsall))
    for instance_ in instances:
        instance, text, class_ = instance_
        features = next(topicsall)
        features.append(class_)
        instance.extend(features)
        data_feat.append(instance)
        count+=1
        
        if(count % 100 == 0):
            print("Done: ",count)

    dataset = {
    'description': 'data_file',
    'relation': 'statements',
    'attributes': attributes,
    'data': data_feat
    }
      
    filename_data = "dataset_topic"+".arff"  
    f = open(filename_data,'w')
    f.write(arff.dumps(dataset))
    f.close()
    
        
    print("Completed")


def generate_arff(extractors, split_arff=True):
    init()
    print("Testing: ")
    test_text = "1.5 million jobs created during the worst economic time this country has had since the Great Depression while the rest of the country lost 400,000 jobs."
    for k,v in zip(get_feature_names(extractors), get_feature_val(extractors, test_text, 0)):
        print(k,v)


    print("Starting: ")

    instances = read_tsv.get_instance()
    colnames = next(instances)
    colnames = [(c, 'STRING') for c in colnames]
    attributes = get_feature_names(extractors)
    colnames.extend(attributes)
    attributes = colnames

    count = 0
    part = 1
    data_feat = []
    for instance_ in instances:
        instance, text, class_ = instance_
        features = get_feature_val(extractors, text, class_)
        instance[3] = instance[3].replace(',', ";")
        instance.extend(features)
        data_feat.append(instance)
        count+=1
        if(count % 100 == 0):
            print("Done: ",count)

        if(split_arff):
            if(count % 500 == 0):
                dataset = {
                'description': 'data_file',
                'relation': 'liwc',
                'attributes': attributes,
                'data': data_feat
                }
                  
                filename_data = "liwc/dataset_"+str(part)+".arff"  
                f = open(filename_data,'w')
                f.write(arff.dumps(dataset))
                f.close()
                part += 1
                data_feat = []
    
    dataset = {
            'description': 'data_file',
            'relation': 'entity',
            'attributes': attributes,
            'data': data_feat
            }

    print(len(data_feat))
              
    if(split_arff):
        filename_data = "entity/dataset_"+str(part)+".arff"  
    else:
        filename_data = "entity/dataset.arff"  
    f = open(filename_data,'w')
    f.write(arff.dumps(dataset))
    f.close()


        
    print("Completed")




if __name__ == '__main__':
    import sys
    module = sys.argv[-1]
    if(module == "all"):
        generate_arff(extractors_)
    elif(module == "topic"):
        feature_topic()
    elif(module == "liwc"):
        generate_arff([liwc], False)
    elif(module == "entity"):
        generate_arff([entity_type], False)
    else:
        print("No such module existss")