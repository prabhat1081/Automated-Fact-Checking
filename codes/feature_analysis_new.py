#from sklearn.neural_network import MLPClassifier
import ast
import numpy as np
import scipy
import pickle
import os

from collections import Counter

from time import time
from collections import defaultdict
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


#from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel


from imblearn.under_sampling import (EditedNearestNeighbours, RandomUnderSampler,
                                     RepeatedEditedNearestNeighbours)
from imblearn.ensemble import EasyEnsemble
from imblearn.pipeline import Pipeline as im_Pipeline

import rank_scorers
import sampler
import feature_importance
import useClaimBuster
import dataset_utils


basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"
workingdir = basepath + "/output_all"

import seaborn as sns
sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()

names = [
"KNN", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis", 
         "MLP"]


classifiers = [
    KNeighborsClassifier(weights='distance', n_neighbors=121),
    SVC(kernel="linear", C=1, probability=True),
    SVC(C=1, probability=True),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage="auto"),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,75,50,25,15), max_iter=10000, random_state=1)
    ]


evaluation_names = ["Accuracy","F1 Score","F1_Micro","F1_Macro","F1_Weighted","Log_Loss","Precision","Recall","ROC_AUC"]

evaluation_methods = []

def evaluate(y_true,y_pred):
	return [accuracy_score(y_true, y_pred),
	f1_score(y_true, y_pred, average="binary"),
	#f1_score(y_true, y_pred, average='micro'),
	#f1_score(y_true, y_pred, average='macro'),
	#f1_score(y_true, y_pred, average='weighted'),
	#log_loss(y_true,y_pred),
	precision_score(y_true, y_pred, average="binary"),
	recall_score(y_true, y_pred, average="binary"),
	roc_auc_score(y_true, y_pred)]



def load_dataset(trainfilelist, indexlist):
    x = []
    Y = []
    allfeatures = []
    embed_feats = []
    allindex = []
    names = []
    for i,files in enumerate(trainfilelist):
        f1 = open(files[0], "r")
        f3 = open(files[1], 'r')
        f2 = open(indexlist[i], "r")
        names = f1.readline()
        names = names.strip().split(" ")[:-1]
        # names = names[:60]
        for lines in f1:
            features = [float(value) for value in lines.split(' ')]
            # features = features[:60] + [features[-1]]
            # print(features)
            allfeatures.append(features)
        for lines in f2:
            indexes = [int(value) for value in lines.split(' ')]
            allindex.append(indexes)
        for lines in f3:
            embeds = [float(value) for value in lines.split(" ")]
            embed_feats.append(embeds)


    # from random import shuffle
    # shuffle(allfeatures)
    n = ["embed"+str(i) for i in range(300)]
    n.extend(names)
    print(len(allfeatures[0]))

    for embeds,feature in zip(embed_feats, allfeatures):
        f = []
        f.extend(embeds)
        f.extend(feature[:-1])
        x.append(f)
        #print(feature[-1])
        Y.append(feature[-1])
    # print(len(names),len(feature))
    # print(Y.count(1))
    # exit(0)
    return n,x, Y, allindex


def normalize_topic_values(X, y):
    X[X<1e-4] = 0
    return X,y



def split_data(X,y, index, val_frac = 0.1, frac=0.2):
    from collections import Counter
    from sklearn.utils import shuffle
    import random
    c = Counter()

    n = len(X)

    X=np.asarray(X)
    y = np.asarray(y)
    index = np.asarray(index)



    for i in range(n):
        if(y[i] == 1):
            c[index[i][0]] += 1;
    l = list(c.items())
    l = shuffle(l, random_state=101)

    test_debates = []

    test_size = int(frac* sum(y))

    k = 0
    while(test_size > 0):
        test_debates.append(l[k][0])
        test_size -= l[k][1]
        k +=1

    print(test_size, test_debates)

    val_debates = []

    val_size = int(val_frac * sum(y))

    
    while(len(val_debates) < 2):
        val_debates.append(l[k][0])
        val_size -= l[k][1]
        k +=1

    print(val_size, val_debates)

    X_test = []
    y_test = []
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    index_test = []
    index_train = []
    index_val = []


    for i in np.random.permutation(n):
        if(index[i][0] in test_debates):
            X_test.append(X[i])
            y_test.append(y[i])
            index_test.append(index[i])
        elif(index[i][0] in val_debates):
            X_val.append(X[i])
            y_val.append(y[i])
            index_val.append(index[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
            index_train.append(index[i])

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    index_test = np.asarray(index_test)
    index_train = np.asarray(index_train)
    index_val = np.asarray(index_val)


    print(np.shape(X_train))


    p = np.random.permutation(len(X_train))
    test_p = np.random.permutation(len(X_test))
    val_p = np.random.permutation(len(X_val))

    return X_train[p], X_test[test_p], y_train[p], y_test[test_p], X_val[val_p], y_val[val_p],  index_train[p], index_test[test_p], index_val[val_p]



def evaluate(X_tests, y_test, index_test, clf, name, sent_print=True):

    y_hats = []
    for i,clfi in enumerate(clf):
        y_hats.append(clfi.predict_proba(X_tests[i])[:,1])
    y_hats = np.asarray(y_hats)
    y_prob = np.mean(y_hats, axis=0)

    #y_hat = np.zeros(np.shape(y_prob)[0])

    y_hat = np.rint(y_prob)




    #y_hat = clf.predict(X_test)
    report = metrics.classification_report(y_test, y_hat)
    #print(str(score))
    f = open(working_dir + "/" + name + '_report.txt', 'w')
    f.write(name+"\n")
    f.write(report)
    print(report)


    # try:
    #     y_prob = clf.predict_proba(X_test)[:,1]
    # except:
    #     pass

    ks = [10,20,30,40,50,60,70,80,90,100,200,300,500,1000]

    allscores = rank_scorers.all_score(y_test, y_prob, ks)

    
        
    for i,k in enumerate(ks):
        print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t", file=f)

    #print(allscores)

    if(not sent_print):
        return
    sent_list = [dataset_utils.get_sentence(idx) for idx in index_test]
    ff = open(working_dir + "/" + name + '_scores.txt', 'w')
    for score, tag, sent in sorted(zip(y_prob, y_test, sent_list)):
        print(tag, score, sent, sep="\t", file=ff)

    # buster_prob = dataset_utils.get_buster_score(index_test)

    # allscores_buster = rank_scorers.all_score(y_test, buster_prob, ks)

    # # for tag, score, sent in zip(y_test, buster_prob, sent_list):
    # #     print(tag, score, sent, sep="\t")
    # print("ClaimBuster",file=f)
    # for i,k in enumerate(ks):
    #     print(k,round(allscores_buster[i][0],3),round(allscores_buster[i][1],3),round(allscores_buster[i][2],3), sep="\t", file=f)




sel_classifiers = [
    SVC(kernel="linear", C=1, probability=True)
    ]

    # SVC(kernel="linear", C=1, probability=True),
    # SVC(C=1, probability=True),
    # RandomForestClassifier(n_estimators=20).
    # AdaBoostClassifier(n_estimators=10)
    
    # ]

sel_names = ["lsvm"]
#sel_names = [ "ada", "lsvm", "rsvm", "rfc"]


def log_scale_counts(X):
    startindex= 366
    endindex = 410

    X[:, startindex:endindex+1] = np.log2(1 + X[:, startindex:endindex+1])
    return X

def main(working_dir, args):
    f_names, X,y, index = load_dataset([(workingdir+"/features.ff", workingdir+"/embeddings.txt")], [workingdir+"/index.txt"])

    print(len(X), len(y))    

    
    X = np.asarray(X)

    y = np.asarray(y)
    index = np.asarray(index)
    f_names = np.asarray(f_names)
    start = 300
    X_part, y = normalize_topic_values(X[:,start:],y)

    X[:,start:] = X_part[:,:]



    #X = log_scale_counts(X)
    print(np.shape(X), np.shape(f_names))
    print(X[0])

    # sel_feats = np.asarray(list(range(0,300)))# + list(range(413,414)))
    sel_feats = np.asarray(list(range(len(X[0]))))
    #sel_feats = np.asarray(range(300))
    X_posonly = X[:,sel_feats]

    print(np.shape(X_posonly))
    f_names = f_names[sel_feats] 

    print(f_names)

      
    X_train, X_test, y_train, y_test, X_val, y_val, index_train, index_test, index_val = split_data(X_posonly, y, index)


    # # pca = PCA(n_components=100)
    # # X_train = pca.fit_transform(X_train)
    
    # print(np.shape(X_train))

    # X_test = pca.transform(X_test)
    # X_val = pca.transform(X_val)

   
    rsampler = RandomUnderSampler(random_state=463)
    

    X_train, y_train = rsampler.fit_sample(X_train, y_train)

    # X_train[:,300:] = scaler.fit_transform(X_train[:,300:])
    # X_test[:,300:] = scaler.transform(X_test[:,300:])
    # X_val[:,300:] = scaler.transform(X_val[:,300:])




   
    for name, clf in zip(sel_names, sel_classifiers):
        print(name)
        name += "comb"

        try:
            raise Exception('Retrain')
            with open(working_dir + "/comb" + name + 'enb.pkl', 'rb') as f1:
                clf1, clf2 = pickle.load(f1)
            # with open(working_dir + "/" + name + '.pkl', 'rb') as f1:
            #     clf2 = pickle.load(f1)
        except:
            #scores = cross_val_score(clf, X_train, y_train, cv=4, scoring="roc_auc")
            #rec_scores = cross_val_score(clf, X_train, y_train, cv=4, scoring="roc_auc")
            #print("Name %s ROC_AUC: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))
            clf1 = SVC(C=1, kernel="linear", probability=True)
            clf2 = SVC(C=1, kernel="linear", probability=True)
            clf1.fit(X_train[:,:300], y_train)
            clf2.fit(X_train[:,300:], y_train)
            with open(working_dir + "/comb" + name + '.pkl', 'wb') as f1:
                pickle.dump((clf1,clf2), f1)  
        evaluate([X_val[:,:300], X_val[:,300:]], y_val, index_val, [clf1,clf2], name)
        evaluate([X_test[:,:300], X_test[:,300:]], y_test, index_test, [clf1,clf2], name+"test")       
    


if __name__ == '__main__':
    import os
    import sys


    working_dir = workingdir+"/models_new" #os.argv[-1]
    try:
        os.makedirs(working_dir)
    except:
        pass

    arguments = sys.argv[1:]
    args = defaultdict(None)
    for x in arguments:
        x = x.split("=")
        args[x[0].strip("-")] = x[1]

    main(working_dir, args)
