#from sklearn.neural_network import MLPClassifier
import ast
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel


from imblearn.under_sampling import (EditedNearestNeighbours, RandomUnderSampler,
                                     RepeatedEditedNearestNeighbours)


basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"
workingdir = basepath + "/new_data_results"

import seaborn as sns
sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

'''
names = ["Decision Trees", "Neural Networks"]


classifiers = [
    DecisionTreeClassifier(),
    MLPClassifier(algorithm='adam', alpha=1e-5, hidden_layer_sizes=(15, 15), random_state=1, verbose=True)
    ]'''

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



def load_dataset(trainfilelist):
    x = []
    Y = []
    allfeatures = []

    for files in trainfilelist:
        f = open(files, 'r')
        for lines in f:
            features = [float(value) for value in lines.split(' ')]
            #print(features)
            allfeatures.append(features)
    # from random import shuffle
    # shuffle(allfeatures)
    print(len(allfeatures[0]))
    for feature in allfeatures:
        x.append(feature[:-1])
        #print(feature[-1])
        Y.append(feature[-1])
    return x, Y


def feature_select(X,y):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape) 
    return X_new, y

def main(working_dir):
    X,y = load_dataset([workingdir+"/features.f"])

    #workingdir = working_dir

    print(len(X), len(y))

    X = np.asarray(X)
    y = np.asarray(y)

    # Instanciate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=2)

    # Fit and transform x to visualise inside a 2D feature space
    X_vis = pca.fit_transform(X)
    # Apply Nearmiss 1
    nm1 =  RandomUnderSampler()
    X_resampled, y_resampled = nm1.fit_sample(X, y)

    print(len(X_resampled), len(y_resampled))
    X_res_vis = pca.transform(X_resampled)

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)
    print(len(X_vis), len(y))
    print(X_vis[y == 0,0])

    ax1.scatter(X_vis[y == 0,0], X_vis[y == 0,1], label="Class #0", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
    ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[4], linewidth=0.15)
    ax1.set_title('Original set')

    ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
                label="Class #0", alpha=.5, edgecolor=almost_black,
                facecolor=palette[0], linewidth=0.15)
    ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
                label="Class #1", alpha=.5, edgecolor=almost_black,
                facecolor=palette[4], linewidth=0.15)
    ax2.set_title('NearMiss 1')

    plt.show()

    X,y = X_resampled, y_resampled

    X, y = feature_select(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    #X_train, X_test, y_train, y_test = X, X, y, y

    '''for name, clf in zip(names, classifiers):
        scores = cross_val_score(clf, X, y, cv=10, scoring="f1")
        print("Name %s Accuracy: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))


    '''
    # total = len(X)
    # train_cnt = int(0.7*total)
    # X_train, y_train = X[:train_cnt], y[:train_cnt]
    # X_test,y_test = X[train_cnt:], y[train_cnt:]
    #print(y_test)
    for name, clf in zip(names, classifiers):
        print(name)
        coll = []
        try:
            with open(working_dir + "/" + name + '.pkl', 'rb') as f1:
                clf = pickle.load(f1)
        except:
            clf.fit(X_train, y_train)
            with open(working_dir + "/" + name + '.pkl', 'wb') as f1:
                pickle.dump(clf, f1)

        coll = clf.predict(X_test)
        #print(coll[10:50], y_test[10:50])
        # for vals in X_test:
        #     z = clf.predict([vals])

        #     coll.append(z[0])
        report = metrics.classification_report(y_test, coll)
        #print(str(score))
        f = open(working_dir + "/" + name + '.txt', 'w')
        f.write(name+"\n")
        f.write(report)
        '''
    for name, clf in zip(names, classifiers):
        print(name)
        coll = []
        clf_selected = Pipeline([
          ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
          ('classification', clf)
        ])
        try:
            with open(working_dir + "/" + name + '_sel.pkl', 'rb') as f1:
                clf = pickle.load(f1)
        except:
            clf.fit(X_train, y_train)
            with open(working_dir + "/" + name + '_sel.pkl', 'wb') as f1:
                pickle.dump(clf, f1)

        coll = clf.predict(X_test)
        #print(coll[10:50], y_test[10:50])
        # for vals in X_test:
        #     z = clf.predict([vals])

        #     coll.append(z[0])
        report = metrics.classification_report(y_test, coll)
        #print(str(score))
        f = open(working_dir + "/" + name + '_sel.txt', 'w')
        f.write(name+"_sel\n")
        f.write(report)
    '''


if __name__ == '__main__':
    import os
    working_dir = workingdir+"/models_balanced1" #os.argv[-1]
    try:
        os.makedirs(working_dir)
    except:
        pass
    main(working_dir)
