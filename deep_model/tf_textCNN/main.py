import numpy as np
from time import time
import os
import pickle
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from data_utils import (train_W2V, load_W2V, build_word2idx_embedMatrix, make_X_train_idx,
                        make_y_train_oneHot, evaluate_matrix)
from model import textCNN_train_test, MAX_SEQ_LEN, in_path
from time import time



if __name__ == '__main__':
    print('——————————————load data——————————————')
    (X_train, y_train), (X_test, y_test) = imdb.load_data()
    X_all = (list(X_train) + list(X_test))[0: ]
    y_all = (list(y_train) + list(y_test))[0: ]
    print(len(X_all), len(y_all))

    imdb_word2idx = imdb.get_word_index()
    imdb_idx2word = dict((idx, word) for (word, idx) in imdb_word2idx.items())
    X_all = [[imdb_idx2word.get(idx - 3, '?') for idx in sen][1:] for sen in X_all]

    w2vModel = train_W2V(X_all, in_path + 'w2vModel')
    word2idx, embedMatrix = build_word2idx_embedMatrix(w2vModel)  # 制作word2idx和embedMatrix

    X_all_idx = make_X_train_idx(X_all, word2idx, MAX_SEQ_LEN)
    y_all_idx = np.array(y_all)  # 一定要注意，X_all和y_all必须是np.array()类型，否则报错
    X_tra_idx, X_val_idx, y_tra_idx, y_val_idx = train_test_split(X_all_idx, y_all_idx, test_size=0.2,
                                                                  random_state=0, stratify=y_all_idx)
    y_tra_oneHot = make_y_train_oneHot(y_tra_idx)
    y_val_oneHot = make_y_train_oneHot(y_val_idx)


    print('——————————————模型的训练和预测——————————————')
    start = time()
    model = textCNN_train_test(embedMatrix)
    model.train([X_tra_idx, y_tra_oneHot])  #不知道为什么，验证非常非常慢！！但keras非常快，很奇怪！！！
    y_pred_idx = model.test([X_val_idx, y_val_oneHot])


    print('——————————————结果评估——————————————')
    y_val_idx = [list(oneHot).index(1) for oneHot in y_val_oneHot]
    evaluate_matrix(y_val_idx, y_pred_idx)

    print(time() - start)


