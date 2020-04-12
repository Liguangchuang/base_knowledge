from keras.preprocessing import sequence
import gensim
import numpy as np






def load_data(file):
    X_context = []
    X_target = []
    y = []

    with open(file, 'rt', encoding='utf-8') as f:
        rawText = f.readlines()
        while rawText:
            line_3 = rawText[0: 3]
            rawText = rawText[3:]

            X_context.append(line_3[0].rstrip().split())
            X_target.append(line_3[1].rstrip().split())

            y_tmp = int(line_3[2].rstrip())
            if y_tmp == -1:
                y_tmp = 0
            elif y_tmp == 0:
                y_tmp = 1
            elif y_tmp == 1:
                y_tmp = 2
            y.append(y_tmp)

    return X_context, X_target, y


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


def make_X_train_idx(X_train, word2idx, maxLen):
    X_train_idx = [[word2idx.get(w, 0) for w in sen] for sen in X_train]
    X_train_idx = np.array(sequence.pad_sequences(X_train_idx, maxlen=maxLen))  # 必须是np.array()类型
    return X_train_idx


def make_y_train_oneHot(y_train_idx):
    y_train_oneHot = np.zeros([len(y_train_idx), len(set(y_train_idx))])
    for i in range(0, len(y_train_oneHot)):
        y_train_oneHot[i][y_train_idx[i]] = 1

    return y_train_oneHot