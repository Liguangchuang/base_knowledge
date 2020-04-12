import numpy as np
import pandas as pd

from keras.engine import Layer  # 非常非常重要，用于添加自定义的层，如attention
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Embedding, LSTM, GRU, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Sequential, load_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras_contrib.layers.crf import CRF

import tensorflow as tf

learn = tf.contrib.learn

import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

##制作训练数据############################
'''
X_train ==[['the','file',...], ['I','love'...],...]
y_train = ['cat', 'dog', 'cat', 'pen', 'ele']
word2idx == {_stopWord:0, love:1, I:2, my:3, you:4, friend:5, my:6}
              [0,       0]
              [0.3,   0.1]
              [-0.4, -0.5]
embedMatrix = [0.5,   0.2] 
              [-0.7,  0.6]       
              [-0.3, -0.8]
              [0.5,   0.2]
'''


##高级操作，“生成器”做词语料库；不清楚为什么用函数写不行
class W2V_corpus_iter():
    def __init__(self, X_train):
        self.X_train = X_train

    def __iter__(self):
        for paragraph in self.X_train:
            for sent in paragraph:
                yield sent   #写成生成器的好处是节省内存！


def train_W2V(W2V_corpus, file):  
    w2vModel = gensim.models.Word2Vec(sentences=W2V_corpus, iter=3, size=300, hs=0, negative=5, min_count=5, window=5)
    w2vModel.save(file)
    return w2vModel


def load_W2V(file):
    # 从gensim那里加载
    try:
        gensim_w2vModel = gensim.models.word2vec.Word2Vec.load(file)
    except:
        gensim_w2vModel = gensim.models.KeyedVectors.load_word2vec_format(file, binary=False)
    ###转化为标准的dict格式
    dict_w2vModel = {}
    for w, g_v in gensim_w2vModel.wv.vocab.items():  #g_v是gensim类型的vocab，别管他！
        dict_w2vModel[w] = gensim_w2vModel[w]

    return dict_w2vModel


def load_W2V(file, word_set=None):  # 跟别人训练好的txt文件里加载
    dict_w2vModel = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip().split(' ')
            w = line[0]
            v = [float(e) for e in line[1:]]
            if word_set != None and w not in word_set:  #用于过滤掉不需要的词，否则会内存爆炸的
                continue
            dict_w2vModel[w] = v

    return dict_w2vModel


def make_embedMatrix(dict_w2vModel):
    ###制作word2idx, embedMatrix
    word2idx = {"_stopWord": 0}  # 这里加了一行是用来过滤停用词的。
    vocab_list = [(w, v) for w, v in dict_w2vModel.items()]
    embedMatrix = np.zeros((len(dict_w2vModel) + 1, len(vocab_list[0][1])))

    for i in range(0, len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embedMatrix[i + 1] = vocab_list[i][1]

    return word2idx, embedMatrix


def pad_seqences(seqences, maxLen, pad_mark=0):  #还是自己写比较靠谱，知道内部做了什么事情！！
    pad_seqences_idx, seqences_len = [], []
    for seq in seqences:
        seqences_len.append(len(seq))
        pad_seqences_idx.append(seq + [pad_mark] * (maxLen - len(seq)))

    return pad_seqences_idx, seqences_len


def sequences2idx(sequences, char2idx):
    sequences_idx = [[char2idx.get(char, 0) for char in seq] for seq in sequences]
    return sequences_idx
	

def make_y_train_oneHot(y_train_idx, cate_dict=None):
    ###如果有字典，就将他转化一下
    if cate_dict != None:
        for i in range(len(y_train_idx)):
            y_train_idx[i] = cate_dict[y_train_idx[i]]
    # 用于检查，是否格式满足要求了
    y_set = set(y_train_idx)
    if y_set != set(range(len(y_set))):
        raise KeyError
    ###正式one-hot化
    y_train_oneHot = np.zeros([len(y_train_idx), len(set(y_train_idx))])
    for i in range(0, len(y_train_oneHot)):
        y_train_oneHot[i][y_train_idx[i]] = 1

    return np.array(y_train_oneHot)
    
    
def bs_seq_lens(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))  #自己写个脚本，试一下就知道了，并不复杂啊！！
    seq_len = tf.reduce_sum(used, reduction_indices=1)
	
    return tf.cast(seq_len, tf.int32)  #返回一个bs中，每条数据的真实长度，


#一个bs的所有句子，真实的序列长度是不一样的，但会通过padding补齐为一样，
#dynamic_rnn通过sequence_length参数给定每句话的长度，当计算到真实长度的时候，就终止。
#当然，如果使用默认的None，就不会终止计算，会计算到最大长度
output, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=lstm_input, initial_state=init_state,   
                                        sequence_length=bs_seq_lens(lstm_input))
'''
例如：一个bs的数据为：
sequences = [[[ 1.53146552 -0.34977303]
             [-1.3270481  -0.06612363]
             [ 0.82136426 -0.10063222]
             [ 0.36684614 -0.76191731]]

            [[-0.98302721  1.50052475]
             [ 0.05744667  0.28317837]
             [ 0.          0.        ]
             [ 0.          0.        ]]

            [[ 0.32490785  0.85719121]
             [ 1.0687954  -0.90649361]
             [ 1.41574305  1.12641253]
             [ 0.19885942 -1.44899067]]]
  
执行length(sequences)；   输出为][4, 2, 4]
因为第二个是用0补齐的句子，实际的长度是2
'''



def _batch_yield(self, X, y, bs, is_shuffle=False):
	data_len = len(X)
	n_batch = int(data_len / bs) if data_len % bs == 0 else int(data_len / bs) + 1

	if is_shuffle:
		idx_shuffle = np.random.permutation(np.arange(data_len))
		X = np.array([X[i] for i in idx_shuffle])
		y = np.array([y[i] for i in idx_shuffle])
		
	for i_bat in range(n_batch):
		sta_idx = i_bat * bs
		end_idx = min((i_bat + 1) * bs, data_len)
		yield X[sta_idx: end_idx], y[sta_idx: end_idx], i_bat



##keras模型的训练、测试代码###############
model = Lstm_model(embedMatrix)
model.fit(X_tra_idx, y_tra_idx, validation_data=(X_val_idx, y_val_idx),
          epochs=1, batch_size=100, verbose=1)
y_pred = model.predict(X_val_idx)
y_pred_idx = [1 if prob[0] > 0.5 else 0 for prob in y_pred]

print(f1_score(y_val_idx, y_pred_idx))
print(confusion_matrix(y_val_idx, y_pred_idx))

##keras保存模型与加载模型：
model.save(inPath + 'model.h5')
load_model(inPath + 'model.h5')  #加载整个模型！
model.load_weights(inPath + 'model.h5')  #加载模型的参数

