import numpy as np

def sigmoid(z):
    return 1.0 / (1 + exp(-z))
    
def gradientDescent(X, Y, alpha):
        n_sample = len(X)
        n_feat = len(X[0])

        one = np.ones((n_sample, 1))  #主要是为了优化b
        X_tmp = np.hstack(X, one)
        
        stop_num = 1
        maxChange = stop_num * 2
        
        W = np.random.randn(1, n_feat+1)
        
        while maxChange > stop_num :
            lastW = W.copy()
            
            #梯度下降更新参数
            for feat_j in range(0, n_feat+1):
                sum_ = 0
                F = np.zeros(n_sample)
                for sam_i in range(0, n_sample):
                    F[sam_i] = sigmoid(X_tmp[sam_i] * W)
                    sum_ += (F[sam_i] - Y[sam_i]) * X_tmp[sam_i][feat_j]

                W[feat_j] = W[feat_j] - (alpha / n_sample) * sum_
            
            #计算是否达到停止条件
            maxChange = 0
            for e in (W - lastW):
                if abs(e) > maxChange:
                    maxChange = abs(e)
    return W
                    
                    

class LogisticRegression():
    def __init__(self, alp=0.2):
        self.alpha = alp
    
    def fit(self, dataSet):
        n_col = len(dataSet[0])
        X = dataSet[:, 0: n_col-1]
        Y = dataSet[:, -1]
        
        self.W = gradientDescent(X, Y, self.alpha)
    
            
    def predict(self, X):       
        Y_pred = np.zeros(len(X))
        
        for col_idx in range(0, len(X)):
            X_vec = X[col_idx]
            y = sigmoid(np.dot(X_vec, self.W))
            Y_pred[col_idx] = y
            
        return Y_pred
