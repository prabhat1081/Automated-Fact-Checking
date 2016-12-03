import os
import numpy  as np

import matplotlib.pyplot as plt

from sklearn import decomposition



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



last = 608

basepath = "/home/bt1/13CS10060/btp"

filename = os.path.join(basepath, "output_all", "embeddings.txt")

embedfile = open(filename)





i = 0
X = []
y = []
for line in embedfile:

	embdes = list(map(float ,line.strip().split(" ")))
	
	X.append(embdes)
	if(i < last):
		y.append(1)
	else:
		y.append(0)

	i += 1

X = np.asarray(X)
y = np.asarray(y)

X_train = X

pca = decomposition.PCA(n_components=2)
pca.fit(X_train)
X_train = pca.transform(X_train)

plot2D(X_train,y)




