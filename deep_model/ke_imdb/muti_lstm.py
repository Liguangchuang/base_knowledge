import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, LSTM, Dropout, Dense, Input
from keras.models import Model
from keras.datasets import imdb
import pickle
import os

from data_utils import (train_W2V, build_word2idx_embedMatrix, make_X_train_idx,
                        make_y_train_oneHot, evaluate_matrix, MAX_SEQ_LEN, in_path)



def muti_Lstm_model(embedMatrix):  # 以后用更易扩展的function模型
    input_layer = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
    embedding_layer = Embedding(input_dim=len(embedMatrix), output_dim=len(embedMatrix[0]),
                                weights=[embedMatrix], trainable=False)(input_layer)

    Lstm_layer_1 = LSTM(units=128, return_sequences=True)(embedding_layer)
    Dropout_layer_1 = Dropout(0.2)(Lstm_layer_1)

    Lstm_layer_2 = LSTM(units=128, return_sequences=True)(Dropout_layer_1)
    Dropout_layer_2 = Dropout(0.2)(Lstm_layer_2)

    Lstm_layer_3 = LSTM(units=128, return_sequences=False)(Dropout_layer_2)
    Dropout_layer_3 = Dropout(0.2)(Lstm_layer_3)

    Dense_layer = Dense(units=2, activation="softmax")(Dropout_layer_3)

    model = Model(inputs=[input_layer], outputs=[Dense_layer])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    print('————————————————load data————————————————')
    (X_train, y_train), (X_test, y_test) = imdb.load_data()
    X_all = (list(X_train) + list(X_test))
    y_all = (list(y_train) + list(y_test))
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
    model = muti_Lstm_model(embedMatrix)
    model.fit(X_tra_idx, y_tra_oneHot, validation_data=(X_val_idx, y_val_oneHot),
              epochs=10, batch_size=500, verbose=1)
    model.save(in_path + 'model.h5')

    y_pred = model.predict(X_val_idx)
    y_pred_idx = [1 if prob[0] > 0.5 else 0 for prob in y_pred]

    print('——————————————结果评估——————————————')
    y_val_idx = [list(oneHot).index(1) for oneHot in y_val_oneHot]
    evaluate_matrix(y_val_idx, y_pred_idx)