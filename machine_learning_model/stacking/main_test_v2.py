from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_curve, auc
from ensemble_learning_v2 import StackingModel_v2
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


def getAuc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    aucs = auc(fpr, tpr)
    return aucs


if __name__ == '__main__':
    for i in range(100):
        print('----------------------------------第{}次预测-----------------------------------'.format(i+1))
        df_all = pd.read_csv("./p0_data.csv")
        for i in range(1, 20):
            df_all = df_all.append(pd.read_csv("./p{}_data.csv".format(i)), ignore_index=True)
        df_all = df_all.sample(frac=1).copy().fillna(-1).reset_index().drop("time_seq", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(df_all.drop("label", axis=1), df_all["label"],test_size=0.2)

        rf_model = RandomForestClassifier()
        adb_model = AdaBoostClassifier()
        et_model = ExtraTreesClassifier()
        lgb_model = lgb.LGBMClassifier()
        lr_model = LogisticRegression()
        gbdt_model = GradientBoostingClassifier()
        Dt_model = DecisionTreeClassifier()

        rf_model.fit(X_train, y_train)
        adb_model.fit(X_train, y_train)
        et_model.fit(X_train, y_train)
        lgb_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        gbdt_model.fit(X_train, y_train)
        Dt_model.fit(X_train, y_train)
        print('rf_model:', round(metrics.f1_score(y_test, rf_model.predict(X_test), average='weighted'), 4))
        print('adb_model:', round(metrics.f1_score(y_test, adb_model.predict(X_test), average='weighted'), 4))
        print('et_model:', round(metrics.f1_score(y_test, et_model.predict(X_test), average='weighted'), 4))
        print('lgb_model:', round(metrics.f1_score(y_test, lgb_model.predict(X_test), average='weighted'), 4))
        print('lr_model:', round(metrics.f1_score(y_test, lr_model.predict(X_test), average='weighted'), 4))
        print('gbdt_model:', round(metrics.f1_score(y_test, gbdt_model.predict(X_test), average='weighted'), 4))
        print('Dt_model:', round(metrics.f1_score(y_test, Dt_model.predict(X_test), average='weighted'), 4))

        stacking_model = StackingModel_v2(topLayer_model=lr_model,
                                          base_model_list=[rf_model, adb_model, et_model, lgb_model, gbdt_model, Dt_model],
                                          use_probas=False,
                                          average_probas=False)
        stacking_model.fit(X_train, y_train)
        res_1 = metrics.f1_score(y_test, stacking_model.predict(X_test), average='weighted')

        stacking_model = StackingModel_v2(topLayer_model=lr_model,
                                          base_model_list=[rf_model, adb_model, et_model, lgb_model, gbdt_model, Dt_model],
                                          use_probas=True,
                                          average_probas=False)
        stacking_model.fit(X_train, y_train)
        res_2 = metrics.f1_score(y_test, stacking_model.predict(X_test), average='weighted')

        stacking_model = StackingModel_v2(topLayer_model=lr_model,
                                          base_model_list=[rf_model, adb_model, et_model, lgb_model, gbdt_model, Dt_model],
                                          use_probas=True,
                                          average_probas=True)
        stacking_model.fit(X_train, y_train)
        res_3 = metrics.f1_score(y_test, stacking_model.predict(X_test), average='weighted')

        print('stacking_model>> ', '类别:',round(res_1,4), '拼接:',round(res_2,4), '平均:',round(res_3,4))
