import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.engine import Layer  #非常非常重要，用于添加自定义的层，如attention
from keras import backend as K
from keras.layers.wrappers import Bidirectional
from keras.layers import Embedding, LSTM, Dropout, Dense, Input
from keras.models import Model
from keras.preprocessing import sequence
from keras.datasets import imdb
from gensim.models.word2vec import Word2Vec
from keras.models import load_model

from data_utils import (train_W2V, build_word2idx_embedMatrix, make_X_train_idx,
                        make_y_train_oneHot, evaluate_matrix, MAX_SEQ_LEN, in_path)


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], 1),   #w:(embedd_dim, 1)
                                 initializer='uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask = None):   #x:(bs, seq_len, embedd_dim)
        eij = K.tanh(K.dot(x, self.W))   #(bs, seq, 1)
        ai = K.exp(eij)  #(bs, seq, 1)

        a_sum = K.sum(ai, axis=1, keepdims=True)  #(bs, 1, 1)
        weights = ai / a_sum   #(bs, seq, 1)

        weighted_input = x * weights  #(bs, seq_len, embedd_dim)
        out = K.sum(weighted_input, axis=1)  #(bs, embedd_dim)

        return out


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])  #需要准确计算出，输出的维度。



def att_biLstm_model(embedMatrix):  # 注意命名不能和库函数同名，之前命名为LSTM()就出很大的错误！！
    input_layer = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
    embedding_layer = Embedding(input_dim=len(embedMatrix), output_dim=len(embedMatrix[0]),
                                weights=[embedMatrix],  trainable=False)(input_layer)
    biLstm_layer = Bidirectional(LSTM(units=20, return_sequences=True), merge_mode='concat')(embedding_layer)
    attention_layer = Attention()(biLstm_layer)
    drop_layer = Dropout(0.5)(attention_layer)
    dense_layer = Dense(units=2, activation="sigmoid")(drop_layer)

    model = Model(inputs=[input_layer], outputs=[dense_layer])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    print('————————————————load data————————————————')
    (X_train, y_train), (X_test, y_test) = imdb.load_data()
    X_all = (list(X_train) + list(X_test))[0: 200]
    y_all = (list(y_train) + list(y_test))[0: 200]
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


    print('————————————————模型的训练预测————————————————')
    model = att_biLstm_model(embedMatrix)
    model.fit(X_tra_idx, y_tra_oneHot, validation_data=(X_val_idx, y_val_oneHot),
              epochs=10, batch_size=500, verbose=1)
    model.save(in_path + 'model.h5')

    y_pred = model.predict(X_val_idx)
    y_pred_idx = [1 if prob[0] > 0.5 else 0 for prob in y_pred]

    print('——————————————结果评估——————————————')
    y_val_idx = [list(oneHot).index(1) for oneHot in y_val_oneHot]
    evaluate_matrix(y_val_idx, y_pred_idx)
