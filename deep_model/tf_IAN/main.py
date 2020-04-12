import numpy as np
import tensorflow as tf
import os
import pickle
from model import IAN_train_test, MAX_SEQ_LEN, in_path
from data_utils import load_data, load_W2V, build_word2idx_embedMatrix
from data_utils import make_X_train_idx, make_y_train_oneHot




if __name__ == '__main__':
    X_tra_c, X_tra_t, y_tra = load_data(in_path+'train.raw')
    X_test_c, X_test_t, y_test = load_data(in_path+'test.raw')
    print(len(X_tra_c), len(X_tra_t), len(y_tra))
    print(len(X_test_c), len(X_test_t), len(y_test))

    if os.path.exists(in_path + 'embedMatrix.pkl'):
        embedMatrix = pickle.load(open(in_path + 'embedMatrix.pkl', 'rb'))
        word2idx = pickle.load(open(in_path + 'word2idx.pkl', 'rb'))
    else:
        all = X_tra_c + X_tra_t + X_test_c + X_test_t
        word_set = set([w for sent in all for w in sent])
        w2vModel = load_W2V(in_path+'glove.42B.300d.txt', word_set=word_set)
        word2idx, embedMatrix = build_word2idx_embedMatrix(w2vModel)
        pickle.dump(embedMatrix, open(in_path + 'embedMatrix.pkl', 'wb'))
        pickle.dump(word2idx, open(in_path + 'word2idx.pkl', 'wb'))

    X_tra_c_idx = make_X_train_idx(X_tra_c, word2idx, MAX_SEQ_LEN)
    X_tra_t_idx = make_X_train_idx(X_tra_t, word2idx, MAX_SEQ_LEN)
    X_test_c_idx = make_X_train_idx(X_test_c, word2idx, MAX_SEQ_LEN)
    X_test_t_idx = make_X_train_idx(X_test_t, word2idx, MAX_SEQ_LEN)

    y_tra_oneHot = make_y_train_oneHot(y_tra)
    y_test_oneHot = make_y_train_oneHot(y_test)

    print('——————————————train model——————————————')
    model = IAN_train_test(embedMatrix)

    model.train([X_tra_c_idx, X_tra_t_idx, y_tra_oneHot])
    y_pred = model.test([X_test_c_idx, X_test_t_idx, y_test_oneHot])
    print(y_pred)
