#from sklearn.neural_network import MLPClassifier
import ast
import numpy as np
import scipy
import pickle
import os


from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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


from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel


from imblearn.under_sampling import (EditedNearestNeighbours, RandomUnderSampler,
                                     RepeatedEditedNearestNeighbours)
from imblearn.ensemble import EasyEnsemble
#from imblearn.pipeline import Pipeline


basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"
workingdir = basepath + "/output_all"

import seaborn as sns
sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()

names = [
"Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis", 
         "MLP"]


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", class_weight="balanced"),
    SVC( class_weight="balanced"),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage="auto"),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,75,50,25,15), max_iter=10000, random_state=1, verbose=True)
    ]


param_grid = {
    "Linear SVM" : {
        'C': scipy.stats.expon(scale=100),
        'gamma': scipy.stats.expon(scale=.1)
    },
    "RBF SVM" : {
        'C': scipy.stats.expon(scale=100),
        'gamma': scipy.stats.expon(scale=.1)
    }
}

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
    names = []
    for files in trainfilelist:
        f = open(files, 'r')
        names = f.readline()
        names = names.strip().split(" ")[:-1]
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
    return names,x, Y


def feature_select(X,y):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape) 
    return X_new, y


def plot_data_and_sample(X,y, sampler):
    # Instanciate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=2)

    # Fit and transform x to visualise inside a 2D feature space
    X_vis = pca.fit_transform(X)

    X_resampled, y_resampled = sampler.fit_sample(X, y)

    print(len(X_resampled), len(y_resampled))


    X_res_vis = []
    for X_res in X_resampled:
        X_res_vis.append(pca.transform(X_res))

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
    ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
    ax1.set_title('Original set')

    ax2.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
    for iy, e in enumerate(X_res_vis):
        ax2.scatter(e[y_resampled[iy] == 1, 0], e[y_resampled[iy] == 1, 1],
                    label="Class #1", alpha=0.5, edgecolor=almost_black,
                    facecolor=np.random.rand(3,), linewidth=0.15)
    ax2.set_title('Easy ensemble')

    plt.show()

    return X_resampled, y_resampled



def plot_ROC_curve(roc_curve):
    false_positive_rate, true_positive_rate, _ = roc_curve
    roc_auc = 0
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_PR_curve(pr_curve):
    precision, recall, _ = pr_curve
    plt.plot(recall, precision, lw=2, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show()



# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



def cross_validate(X,y):
    for name, clf in zip(names[1:3], classifiers[1:3]):
        clf_scaled = Pipeline([
          ('scaling', StandardScaler()),
          ('classification', clf)
        ])
        scores = cross_val_score(clf, X, y, cv=4, scoring="roc_auc")
        print("Name %s ROC_AUC: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))



def randomGridSearch(X,y):
    for name, clf in zip(names[1:3], classifiers[1:3]):
        # run randomized search
        n_iter_search = 2
        random_search = RandomizedSearchCV(clf, param_distributions=param_grid[name],
                                           n_iter=n_iter_search)

        start = time()
        random_search.fit(X, y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(random_search.cv_results_)



def main(working_dir):
    f_names, X,y = load_dataset([workingdir+"/features.ff"])

    print(len(X), len(y), f_names)

    X = np.asarray(X)
    y = np.asarray(y)

    
    randomGridSearch(X,y)

    exit(0)
    #sampler= EasyEnsemble()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)



    pipe_components = [None]

    for name, clf in zip(names[1:3], classifiers[1:3]):
        print(name)
        pipe_components[-1] = ('classification', clf)
        clf = Pipeline(pipe_components)
        try:
            with open(working_dir + "/" + name + '.pkl', 'rb') as f1:
                clf = pickle.load(f1)
        except:
            clf.fit(X_train, y_train)
            with open(working_dir + "/" + name + '.pkl', 'wb') as f1:
                pickle.dump(clf, f1)

        y_hat = clf.predict(X_test)

        report = metrics.classification_report(y_test, y_hat)
        #print(str(score))
        f = open(working_dir + "/" + name + '.txt', 'w')
        f.write(name+"\n")
        f.write(report)
        print(report)
        plot_ROC_curve(metrics.roc_curve(y_test, clf.decision_function(X_test), pos_label=1))
        plot_PR_curve(metrics.precision_recall_curve(y_test,clf.decision_function(X_test), pos_label=1 ))


if __name__ == '__main__':
    import os
    working_dir = workingdir+"/models_balanced_SVM" #os.argv[-1]
    try:
        os.makedirs(working_dir)
    except:
        pass
    main(working_dir)
