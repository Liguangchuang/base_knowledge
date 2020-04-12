import numpy as np
import csv
import os
import pickle
from data_utils import train_W2V, load_W2V, build_word2idx_embedMatrix, make_X_train_idx, split_words
from data_utils import make_y_train_oneHot, evaluate_matrix, load_data, W2V_corpus_iter
from model import HAN_train_test, MAX_SENT_NUM, MAX_SENT_LEN, in_path




if __name__ == '__main__':
    if os.path.exists(in_path + 'yelp_2014_data'):   #返回True 或 False:
        print('load_data')
        yelp_2014_data = pickle.load(open(in_path + 'yelp_2014_data', 'rb'))  # 加载

        X_train_idx = yelp_2014_data['X_train_idx']
        X_test_idx = yelp_2014_data['X_test_idx']
        y_train_oneHot = yelp_2014_data['y_train_oneHot']
        y_test_oneHot = yelp_2014_data['y_test_oneHot']
        embedMatrix = yelp_2014_data['embedMatrix']

        print(len(X_train_idx), len(X_test_idx), len(y_train_oneHot), len(y_test_oneHot))

    else:
        print('preprocess_data')
        X_train, y_train = load_data(in_path + 'yelp-2014-seg-20-20.train.ss')
        X_test, y_test = load_data(in_path + 'yelp-2014-seg-20-20.test.ss')

        X_train = [paragraph.split(' <sssss> ') for paragraph in X_train]
        X_test = [paragraph.split(' <sssss> ') for paragraph in X_test]

        X_train = [[split_words(sent) for sent in paragraph] for paragraph in X_train]
        X_test = [[split_words(sent) for sent in paragraph] for paragraph in X_test]

        W2V_corpus = W2V_corpus_iter(X_train)
        w2vModel = train_W2V(W2V_corpus, in_path + 'w2vModel')
        word2idx, embedMatrix = build_word2idx_embedMatrix(w2vModel)  # 制作word2idx和embedMatrix

        X_train_idx = make_X_train_idx(X_train, word2idx, MAX_SENT_NUM, MAX_SENT_LEN)
        X_test_idx = make_X_train_idx(X_test, word2idx, MAX_SENT_NUM, MAX_SENT_LEN)

        y_train_oneHot = make_y_train_oneHot(y_train, is_cate_dict=True)
        y_test_oneHot = make_y_train_oneHot(y_test, is_cate_dict=True)

        print(len(X_train_idx), len(X_test_idx), len(y_train_oneHot), len(y_test_oneHot))

        yelp_2014_data = {}
        yelp_2014_data['X_train_idx'] = X_train_idx
        yelp_2014_data['X_test_idx'] = X_test_idx
        yelp_2014_data['y_train_oneHot'] = y_train_oneHot
        yelp_2014_data['y_test_oneHot'] = y_test_oneHot
        yelp_2014_data['embedMatrix'] = embedMatrix

        pickle.dump(yelp_2014_data, open(in_path + 'yelp_2014_data', 'wb'))



    print('——————————————模型的训练和预测——————————————')
    model = HAN_train_test(embedMatrix)

    model.train([X_train_idx, y_train_oneHot])
    y_pred = model.test([X_test_idx, y_test_oneHot])
    print(y_pred)



