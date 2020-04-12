import numpy as np
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')


class StackingModel_v1():
    def __init__(self, topLayer_model, base_model_list, n_fold=5, use_probas=True, average_probas=False):
        self.topLayer_model = topLayer_model
        self.base_model_list = base_model_list
        self.n_flod = n_fold  # 默认5折交叉
        self.use_probas=use_probas
        self.average_probas = average_probas

    def fit(self, X_train, y_train, X_test):
        X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)

        for i, model in enumerate(self.base_model_list):
            #print('model_{}'.format(i))
            train_pred = []
            test_pred = []
            for j, (tra_idx, val_idx) in enumerate(KFold(n_splits=self.n_flod).split(X_train)):
                X_tra, X_val = X_train[tra_idx], X_train[val_idx]
                y_tra, _ = y_train[tra_idx], y_train[val_idx]
                model.fit(X_tra, y_tra)
                if self.use_probas:
                    train_pred += model.predict_proba(X_val).tolist()
                    test_pred.append(model.predict_proba(X_test).tolist())
                else:
                    train_pred += [[e]for e in model.predict(X_val)]
                    test_pred.append([[e]for e in model.predict(X_test)])
            train_pred = np.array(train_pred)
            test_pred = np.mean(np.array(test_pred), axis=0)  #测试集可以选择(加权平均)
            if i == 0:
                X_train_stack = train_pred
                self.X_test_stack = test_pred
            else:
                if not self.average_probas:
                    X_train_stack = np.c_[X_train_stack, train_pred]
                    self.X_test_stack = np.c_[self.X_test_stack, test_pred]
                else:
                    #将每个模型的预测，求平均
                    X_train_stack += train_pred
                    self.X_test_stack += test_pred
                    if i == len(self.base_model_list) - 1:
                        X_train_stack = X_train_stack / len(self.base_model_list)
                        self.X_test_stack = self.X_test_stack / len(self.base_model_list)
        
        # 顶层模型的训练
        self.topLayer_model.fit(X_train_stack, y_train)


    def predict(self):  # 测试集的数据是X_test_stack，而不是原来的X_test
        return self.topLayer_model.predict(self.X_test_stack)


    def predict_proba(self):
        return self.topLayer_model.predict_proba(self.X_test_stack)

