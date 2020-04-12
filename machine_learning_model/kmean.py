import numpy as np

def distEucli(vecA, vecB):  
    squareSum = 0
    for idx in range(0, len(vecA)):
        squareSum += (vecA[idx] - vecB[idx]) ** 2
    return np.sqrt(squareSum)

def randGetClusters(dataSet, k):  #默认dataSet是np.array类型
    n_sample = len(dataSet)
    n_feat = len(dataSet[0])
    clusters = np.zeros((1, n_feat))
    
    for _ in range(0, k):
        randIdx = np.random.randint(0, n_sample)
        clusters = np.vstack((clusters, dataSet[randIdx]))
    clusters = clusters[1: ]
    
    return clusters
        
    
def KMean(dataSet, k, getDist = distEucli, getClusters = randGetClusters):
    n_sample = len(dataSet)
    n_feat = len(dataSet[0])
    
    clusters = getClusters(dataSet, k)
    sam_clus = np.zeros((n_sample, 2))
    dist_list = []
    stop_num = 10
    totalChangeDist = stop_num * 2
    
    
    while totalChangeDist > stop_num:
        #所有样本寻找聚类中心
        for sam_idx in range(0, len(dataSet)):
            for clu_idx in range(0, len(clusters)):
                dist_list.append(getDist(dataSet[sam_idx], clusters[clu_idx]))
            clu_label = dist_list.index(min(dist_list))
            sam_clus[sam_idx] = [sam_idx, clu_label]

        #更新聚类中心
        last_clusters = clusters.copy()
        for clu_idx in range(0, len(clusters)):
            count = 0
            sum_vec = np.zeros(n_feat)
            for sam_idx in range(0, len(sam_clus)):
                if sam_clus[sam_idx][1] == clu_idx:
                    count += 1
                    sum_vec += dataSet[sam_idx]
            clusters[clu_idx] = sum_vec / count
        
        #计算停止条件
        totalChangeDist = 0
        for clu_idx in range(0, len(clusters)):
            totalChangeDist += getDist(last_clusters[clu_idx], clusters[clu_idx])
        
    return sam_clus, clusters