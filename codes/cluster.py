from sklearn.cluster import KMeans

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
        print(names[-1])
        names = names.strip().split(" ")[:-1]
        # names = names[:60]
        for lines in f1:
            # features = [float(value) for value in lines.split(' ')]
            features = int(lines.split(' ')[-1])
            # print('loaded class : ', float(lines.split(' ')[-1]))
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
    # n.extend(names)
    # print(len(allfeatures[0]))

    # for embeds,feature in zip(embed_feats, allfeatures):
    #     f = []
    #     f.extend(embeds)
    #     f.extend(feature[:-1])
    #     x.append(f)
    #     #print(feature[-1])
    #     Y.append(feature[-1])
    
    x = embed_feats
    Y = allfeatures

    # print(len(names),len(feature))
    # print(Y.count(1))
    # exit(0)
    return n,x, Y, allindex




def main(working_dir):
    f_names, X,y, index = load_dataset([(workingdir+"/features.ff", workingdir+"/embeddings.txt")], [workingdir+"/index.txt"])

    print(len(X), len(y))
    # print (max(y), type(y), type(y[10]))
    kmeans = KMeans(n_clusters=100,random_state=100).fit(X)
    print(kmeans.cluster_centers_)
    print(len(kmeans.labels_))
    clusters = [[0,0]]*100
    for i in range(len(y)):
        if y[i]==1:
            # print('Yay bitch!')
            # print(clusters[kmeans.labels_[i]], type(clusters[kmeans.labels_[i]][0]))
            clusters[kmeans.labels_[i]]=[clusters[kmeans.labels_[i]][0]+1, clusters[kmeans.labels_[i]][1]]
        else:
            clusters[kmeans.labels_[i]]=[clusters[kmeans.labels_[i]][0], clusters[kmeans.labels_[i]][1]+1]
    for cluster in clusters:
        print(cluster)


    
    # X = np.asarray(X)
    # y = np.asarray(y)
    # index = np.asarray(index)
    # f_names = np.asarray(f_names)
    # start = 300
    # X_part, y = normalize_topic_values(X[start:],y)

    # X[start:] = X_part[:]

    # print(np.shape(X), np.shape(f_names))
    # print(X[0])

    # sel_feats = np.asarray(list(range(0,300)))
    # X_posonly = X[:,sel_feats]

    # print(np.shape(X_posonly))
    # f_names = f_names[sel_feats] 
    # print(f_names)
    # X_train, X_test, y_train, y_test, index_train, index_test = split_data(X_posonly, y, index)

    # pca = PCA(n_components=100)
    # X_train = pca.fit_transform(X_train)
    
    # print(np.shape(X_train))

    # X_test = pca.transform(X_test)

    # X_vis= X_train


basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"
workingdir = basepath + "/output_all"
working_dir = workingdir+"/models_feat/finals_2"
main(working_dir)
