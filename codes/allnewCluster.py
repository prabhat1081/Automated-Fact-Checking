import json
import os



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import extractors.tokenizer
from sklearn import decomposition
from sklearn import datasets
from sklearn import ensemble
from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


from imblearn.under_sampling import RandomUnderSampler

import seaborn as sns
sns.set()


exit(0)
# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()

basepath = "/home/bt1/13CS10060/btp"

filename = os.path.join(basepath, "ayush_dataset", "data.json")


data = json.load(open(filename))

print(len(data['samples']))



def plot3D(X,y):

	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	plt.cla()

	colors = [palette[0], palette[2], palette[5]]

	for color, i, target_name in zip(colors, [0, 1], ["no", "yes"]):
	    ax.scatter(X[y == i, 0], X[y == i, 1], color=color)

	#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)

	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])


	plt.show()


def plot2D(X ,y):

	X = np.asarray(X)
	y = np.asarray(y)
	plt.figure()
	colors = ["blue", "darkorange"]
	lw = 0.15

	for color, i, target_name in zip(colors, [0, 1], ["no", "yes"]):
	    plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.5, lw=lw,
	                label=target_name)


	plt.show()





def dBSCAN(X, labels_true):
	db = DBSCAN(eps=0.3, min_samples=10).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print('Estimated number of clusters: %d' % n_clusters_)
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
	print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
	print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
	print("Adjusted Rand Index: %0.3f"
	      % metrics.adjusted_rand_score(labels_true, labels))
	print("Adjusted Mutual Information: %0.3f"
	      % metrics.adjusted_mutual_info_score(labels_true, labels))
	print("Silhouette Coefficient: %0.3f"
	      % metrics.silhouette_score(X, labels))

tag = []
embeds = []



testid = [3,4,13,14]

X_test = []
y_test = []
y_train = []
X_train = []

for sample in data['samples']:

	tagi = 0
	if(sample['Marked'] == "Y"):
		tag.append(1)
		tagi = 1
	else:
		tag.append(0)

	embeds.append(sample['Embeds'])

	if(int(sample['DebateId']) in testid):
		X_test.append(sample['Embeds'])
		y_test.append(tagi)

	else:
		X_train.append(sample['Embeds'])
		y_train.append(tagi)


print(len(embeds), len(tag), len(embeds[0]))

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
y_train = np.asarray(y_train)

print(len(X_test), len(X_train))
			
p = np.random.permutation(len(X_train))
test_p = np.random.permutation(len(X_test))

X_train, y_train = X_train[p], y_train[p]
X_test, y_test = X_test[test_p], y_test[test_p]

X = embeds

dBSCAN(X, tag)
exit(0)


spectral = cluster.SpectralClustering(n_clusters=2,eigen_solver='arpack')

y_pred = spectral.fit_predict(X)
plot2D(X, y_pred)

print(y_pred)

report = metrics.classification_report(tag, y_pred)

print(report)



#dBSCAN(X, tag)

pca = decomposition.PCA(n_components=100)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


plot2D(X_train, y_train)





rsampler = RandomUnderSampler(random_state=745)
X_train, y_train = rsampler.fit_sample(X_train, y_train)
#X_test, y_test = rsampler.fit_sample(X_test, y_test)





clf = svm.SVC(C=1, kernel="linear")

clf.fit(X_train,y_train)
y_hat = clf.predict(X_test)
report = metrics.classification_report(y_test, y_hat)

print(report)


for i in range(10):
	rsampler = RandomUnderSampler()
	X_test_s, y_test_s = rsampler.fit_sample(X_test, y_test)
	y_hat = clf.predict(X_test_s)

	#plot2D(X_test_s, y_test_s)
	#plot2D(X_test_s, y_hat)
	report = metrics.classification_report(y_test_s, y_hat)

	print(report)












