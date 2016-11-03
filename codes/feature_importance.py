print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier



def plot_feature_importance(X, y, names):

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
               axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
         color="r", yerr=std[indices], align="center")
    ordered_names = [names[ii] for ii in indices]
    plt.xticks(range(X.shape[1]), ordered_names)
    plt.xlim([-1, X.shape[1]])
    plt.show()