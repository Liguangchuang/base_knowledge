import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.preprocessing import sequence
from gensim.models.word2vec import Word2Vec


MAX_SEQ_LEN = 250
in_path = 'D:/python_code/AI_data/'


def train_W2V(sentenList, file):
    w2vModel = Word2Vec(sentences=sentenList, iter=3, size=300, hs=0, negative=5, min_count=5, window=5)
    w2vModel.save(file)
    return w2vModel


def build_word2idx_embedMatrix(w2vModel):
    word2idx = {"_stopWord": 0}  # 这里加了一行是用来过滤停用词的。
    vocab_list = [(w, w2vModel.wv[w]) for w, v in w2vModel.wv.vocab.items()]
    embedMatrix = np.zeros((len(w2vModel.wv.vocab.items()) + 1, w2vModel.vector_size))

    for i in range(0, len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embedMatrix[i + 1] = vocab_list[i][1]

    return word2idx, embedMatrix


def make_X_train_idx(sentenList, word2idx, maxLen):
    X_train_idx = [[word2idx.get(w, 0) for w in sen] for sen in sentenList]
    X_train_idx = np.array(sequence.pad_sequences(X_train_idx, maxlen=maxLen))  # 必须是np.array()类型
    return X_train_idx


def make_y_train_oneHot(y_train_idx):
    y_train_oneHot = np.zeros([len(y_train_idx), len(set(y_train_idx))])
    for i in range(0, len(y_train_oneHot)):
        y_train_oneHot[i][y_train_idx[i]] = 1

    return y_train_oneHot


def evaluate_matrix(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    print('Acc:', accuracy_score(y_true, y_pred))
    print('precision:', precision_score(y_true, y_pred, average='macro'))
    print('recall:', recall_score(y_true, y_pred, average='macro'))
    print('F1:', f1_score(y_true, y_pred, average='macro'))
    return f1_score(y_true, y_pred, average='macro')
