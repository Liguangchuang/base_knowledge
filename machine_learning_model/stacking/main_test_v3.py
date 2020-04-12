from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd


from ensemble_learning_v3 import StackingModel_v3
from sklearn import metrics
import logging
logging.basicConfig(filename='res.log', level=logging.DEBUG,  format="%(asctime)s  - %(levelname)s - %(message)s")


if __name__ == '__main__':
    print("load_data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for _ in range(100):
        df_all = pd.read_csv("./p0_data.csv")
        for i in range(1, 20):
            df_all = df_all.append(pd.read_csv("./p{}_data.csv".format(i)), ignore_index=True)
        df_all = df_all.sample(frac=1).copy().fillna(-1).reset_index().drop("time_seq", axis=1)
        print(df_all.shape)

        df_train = df_all[0: 32753].reset_index()
        df_test = df_all[32753: 32753 + 10917].reset_index()
        df_val = df_all[32753 + 10917:].reset_index()

        X_train = df_train.drop("label", axis=1)
        X_test = df_test.drop("label", axis=1)
        X_val = df_val.drop("label", axis=1)
        y_train = df_train["label"]
        y_test = df_test["label"]
        y_val = df_val["label"]

        rf_model = RandomForestClassifier()
        adb_model = AdaBoostClassifier()
        et_model = ExtraTreesClassifier()
        lgb_model = lgb.LGBMClassifier()
        lr_model = LogisticRegression()
        gbdt_model = GradientBoostingClassifier()
        dt_model = DecisionTreeClassifier()

        stacking_model = StackingModel_v3(topLayer_model=lr_model,
                                          base_model_list=[rf_model, adb_model, et_model, lgb_model, gbdt_model, dt_model],
                                          val_weight_average=False,
                                          )
        stacking_model.fit(X_train, y_train)
        res_1 = metrics.f1_score(y_test, stacking_model.predict(X_test), average='weighted')

        stacking_model = StackingModel_v3(topLayer_model=lr_model,
                                          base_model_list=[rf_model, adb_model, et_model, lgb_model, gbdt_model, dt_model],
                                          val_weight_average=True,
                                          val_set=[X_val, y_val]
                                          )
        stacking_model.fit(X_train, y_train)
        res_2 = metrics.f1_score(y_test, stacking_model.predict(X_test), average='weighted')
        print(res_1, res_2)

        logging.info('普通平均:{}, 加权平均:{}'.format(round(res_1, 4), round(res_2, 4)))
