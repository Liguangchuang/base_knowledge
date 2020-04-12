from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier,ExtraTreesClassifier)
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc


class StackingModel():
    def __init__(self, topLayer_model, base_model_list, n_fold=5):
        self.topLayer_model = topLayer_model
        self.base_model_list = base_model_list
        self.n_flod = n_fold  #默认5折交叉验证


    def fit(self, X_train, y_train, X_test):
        X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)

        #基模型的训练和预测，用于构造X_train_stack和X_test_stack
        X_train_stack = []
        X_test_stack = []
        for i, model in enumerate(self.base_model_list):
            print('model_{}'.format(i))
            train_pred = []
            test_pred = []
            KF = KFold(n_splits=self.n_flod)
            for tra_idx, val_idx in KF.split(X_train, y_train):
                X_tra, X_val = X_train[tra_idx], X_train[val_idx]
                y_tra, y_val = y_train[tra_idx], y_train[val_idx]

                model.fit(X_tra, y_tra)
                train_pred.append(list(model.predict(X_val)))
                test_pred.append(list(model.predict(X_test)))
            
            train_pred = np.array([e for list_ in train_pred for e in list_])
            test_pred = np.mean(test_pred, axis=0)
            X_train_stack.append(train_pred)
            X_test_stack.append(test_pred)

        X_train_stack = np.array(X_train_stack).T
        self.X_test_stack = np.array(X_test_stack).T

        #顶层模型的训练
        self.topLayer_model.fit(X_train_stack, y_train)


    def predict(self):   #测试集的数据是X_test_stack，而不是原来的X_test
        return self.topLayer_model.predict(self.X_test_stack)


def getAuc(y_true,y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    aucs = auc(fpr,tpr)
    return aucs


if __name__ == '__main__':
    print("load_data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    df_all = pd.read_csv("./train_baseline.csv")
    df_all = df_all.sample(frac=1).copy()
    df_all = df_all.fillna(-1)
    df_all = df_all.reset_index()
    print(df_all.shape)

    df_train, df_test = train_test_split(df_all, test_size=0.2)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    X_train, X_test, y_train, y_test = train_test_split(df_train.drop("Tag", axis=1), df_train["Tag"], test_size=0.2)


    print("stacking_model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    rf_model = RandomForestClassifier()
    adb_model = AdaBoostClassifier()
    gdbc_model = GradientBoostingClassifier()
    et_model = ExtraTreesClassifier()

    topLayer_model = lgb.LGBMClassifier()
    base_model_list = [rf_model, adb_model, gdbc_model, et_model]

    stacking_model = StackingModel(topLayer_model, base_model_list)
    stacking_model.fit(X_train, y_train, X_test)
    print('stacking_model:', getAuc(y_test, stacking_model.predict()))


    print("other_model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    rf_model = RandomForestClassifier()
    adb_model = AdaBoostClassifier()
    gdbc_model = GradientBoostingClassifier()
    et_model = ExtraTreesClassifier()

    rf_model.fit(X_train, y_train)
    adb_model.fit(X_train, y_train)
    gdbc_model.fit(X_train, y_train)
    et_model.fit(X_train, y_train)

    print('rf_model:', getAuc(y_test, rf_model.predict(X_test)))
    print('adb_model:', getAuc(y_test, adb_model.predict(X_test)))
    print('gdbc_model:', getAuc(y_test, gdbc_model.predict(X_test)))
    print('et_model:', getAuc(y_test, et_model.predict(X_test)))





'''
终于搞定stacking，太开心了！！！
之前一直以为，这个东西贼他妈神秘；搞懂了，发现贼他妈简单！！！！
'''

