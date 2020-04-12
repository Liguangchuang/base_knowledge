import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Dense, Dropout, Activation, Flatten, concatenate
from keras.models import Model
from keras import optimizers
from keras.datasets import imdb
from keras.layers import Conv1D, MaxPooling1D
import pickle
import os
from time import time

from data_utils import (train_W2V, build_word2idx_embedMatrix, make_X_train_idx,
                        make_y_train_oneHot, evaluate_matrix, MAX_SEQ_LEN, in_path)



def TextCnn_model(embedMatrix):
    comment_seq = Input(shape=[MAX_SEQ_LEN], name='x_seq')  #input_layer

    emb_comment = Embedding(input_dim=len(embedMatrix), output_dim=len(embedMatrix[0]),
                            weights=[embedMatrix], trainable=False)(comment_seq)

    # 卷积层与池化层
    convs = []
    filter_sizes = [2, 3, 4]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(MAX_SEQ_LEN - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.5)(merge)

    # 全连接层
    output = Dense(units=30, activation='relu')(out)  # units:输出维度
    output = Dense(units=2, activation='softmax')(output)

    model = Model([comment_seq], output)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


if __name__ == '__main__':
    print('————————————————load data————————————————')
    (X_train, y_train), (X_test, y_test) = imdb.load_data()
    X_all = (list(X_train) + list(X_test))[0: 1000]
    y_all = (list(y_train) + list(y_test))[0: 1000]
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


    print('————————————————模型的训练和预测————————————————')
    start = time()
    model = TextCnn_model(embedMatrix)
    model.fit(X_tra_idx, y_tra_oneHot,
              epochs=1, batch_size=500, verbose=1)
    model.save(in_path + 'model.h5')

    # validation_data=(X_val_idx, y_val_oneHot),

    y_pred = model.predict(X_val_idx)
    y_pred_idx = [1 if prob[0] > 0.5 else 0 for prob in y_pred]

    print('——————————————结果评估——————————————')
    y_val_idx = [list(oneHot).index(1) for oneHot in y_val_oneHot]
    evaluate_matrix(y_val_idx, y_pred_idx)

    print(time() - start)