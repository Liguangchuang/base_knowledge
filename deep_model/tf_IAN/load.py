import numpy as np
import gensim
import sys

def load_W2V(file):
    try:
        gensim_w2vModel = gensim.models.word2vec.Word2Vec.load(file)
    except:
        gensim_w2vModel = gensim.models.KeyedVectors.load_word2vec_format(file, binary=False)

    dict_w2vModel = {}
    for w, v in gensim_w2vModel.wv.vocab.items():
        dict_w2vModel[w] = v
    return dict_w2vModel


def load_W2V(file, word_set=None):
    print('load by myself')
    f = open(file, encoding='utf-8')
    embedd_dim = len(next(f).rstrip().split()[1:])
    dict_w2vModel = {}
    for line in f:
        line = line.rstrip().split()
        w = line[0]
        v = line[1:]
        if (len(v) == embedd_dim):  #注意，if的判断语句不可交换
            if (word_set == None) or (w in word_set):  #只选择出现的词，避免内存溢出；
                dict_w2vModel[w] = v
    return dict_w2vModel




def build_word2idx_embedMatrix(w2vModel):
    word2idx = {"_stopWord": 0}  # 这里加了一行是用来过滤停用词的。
    vocab_list = [(w, w2vModel[w]) for w, v in w2vModel.items()]
    embedMatrix = np.zeros((len(w2vModel) + 1, len(w2vModel['the'])))

    for i in range(0, len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embedMatrix[i + 1] = vocab_list[i][1]

    return word2idx, embedMatrix


 #以M显示

in_path = '../../../../deep_data/'
w2vModel = load_W2V(in_path+'Tencent500000.txt', 'rt')



#
# f = open(in_path+'Tencent500000.txt', encoding='utf-8')
# embedd_dim = len(next(f).rstrip().split()[1:])
# w2vModel = {}
# for line in f:
#     line = line.rstrip().split()
#     w = line[0]
#     v = line[1:]
#     if len(v) == embedd_dim:
#         w2vModel[w] = v
#         print(len(w2vModel))
#         # print(w2vModel)
#
#
# while True:
#     print(1)
#     pass