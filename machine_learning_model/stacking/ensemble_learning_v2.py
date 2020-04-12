import numpy as np
from sklearn.model_selection import KFold
import copy
import warnings
warnings.filterwarnings('ignore')


class StackingModel_v2:
    def __init__(self, topLayer_model, base_model_list,
                 n_fold=5, use_probas=True, average_probas=False):
        self.topLayer_model = topLayer_model
        self.base_model_list = base_model_list #存储M个输入的未训练模型
        self.n_flod = n_fold  # 默认5折交叉
        self.use_probas = use_probas
        self.average_probas = average_probas

    def fit(self, X_train, y_train):
        X_train, y_train = np.array(X_train), np.array(y_train)
        self.had_train_models = []  # 存储训练好的(M * K)个模型
        for i, model in enumerate(self.base_model_list):
            train_pred = []
            KFold_models = []
            for j, (tra_idx, val_idx) in enumerate(KFold(n_splits=self.n_flod).split(X_train)):
                X_tra, X_val = X_train[tra_idx], X_train[val_idx]
                y_tra, y_val = y_train[tra_idx], y_train[val_idx]
                model.fit(X_tra, y_tra)
                KFold_models.append(copy.deepcopy(model))
                if self.use_probas:
                    train_pred += model.predict_proba(X_val).tolist()
                else:
                    train_pred += [[e]for e in model.predict(X_val)]
            self.had_train_models.append(copy.deepcopy(KFold_models))  #存储训练好的K折模型，用于预测
            train_pred = np.array(train_pred)
            if i == 0:
                X_train_stack = train_pred
            else:
                if not self.average_probas:
                    X_train_stack = np.c_[X_train_stack, train_pred]
                else:
                    #将每个模型的预测，求平均
                    X_train_stack += train_pred
                    if i == len(self.base_model_list) - 1:
                        X_train_stack = X_train_stack / len(self.base_model_list)
        # 顶层模型的训练
        self.topLayer_model.fit(X_train_stack, y_train)

    def predict(self, X_test):
        return self.__predict_tmp(X_test, out_probas=False)

    def predict_proba(self, X_test):
        return self.__predict_tmp(X_test, out_probas=True)

    def __predict_tmp(self, X_test, out_probas=False):  # 测试集的数据是X_test_stack，而不是原来的X_test
        for i, KF_models in enumerate(self.had_train_models):
            test_pred = []
            for model in KF_models:
                if self.use_probas:
                    test_pred.append(model.predict_proba(X_test).tolist())
                else:
                    test_pred.append([[e]for e in model.predict(X_test)])
            test_pred = np.mean(np.array(test_pred), axis=0)  # 测试集可以选择(加权平均)
            if i == 0:
                X_test_stack = test_pred
            else:
                if not self.average_probas:
                    X_test_stack = np.c_[X_test_stack, test_pred]
                else:
                    X_test_stack += test_pred
                    if i == len(self.base_model_list) - 1:
                        X_test_stack = X_test_stack / len(self.base_model_list)
        #顶层模型预测
        if out_probas:
            return self.topLayer_model.predict_proba(X_test_stack)
        else:
            return self.topLayer_model.predict(X_test_stack)



