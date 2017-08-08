def sampling(features, target):
    
    '''This function draws samples from a large dataset by grouping similar data points (KMeans) into n groups and randomly select 1/n*100 data points of each group.'''
    from sklearn.cluster import KMeans
    import math
    
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    n_clusters = int(math.log(len(features), 10))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    
    label = pd.DataFrame(kmeans.labels_)
    sample = pd.concat([features, label], axis=1)
    sample.columns.values[-1] = 'label'
    sample = pd.concat([sample, target], axis=1)
    
    cluster = []
    for i in range(n_clusters+1):
        cluster_ = 'cluster_%d'%i
        cluster_ = sample[sample['label'] ==i]
        cluster.append(cluster_)

    result = pd.DataFrame()
    log_len = math.log(len(features), 10)
    percentage = 1000-int((sigmoid(log_len))*1000)
    perc = 100/percentage
    
    for i in cluster: 
        sub = len(sample)//(n_clusters*perc)
        if i.shape[0] < sub: 
            sub = i.shape[0]
        if i.shape[0] == 0: 
            break
        sub = int(sub)
        idx = np.random.choice(i.shape[0], sub, replace=False)
        result = result.append(i.iloc[idx])


    features = result.iloc[:,0:-2]
    target = result.iloc[:,-1:]

    return features, target