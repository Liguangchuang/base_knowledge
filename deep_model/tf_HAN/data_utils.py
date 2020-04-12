import numpy as np
import csv
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



def load_data(file):
    X_train = []
    y_train = []
    with open(file, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        for row in csv_reader:    #row将每一行读成list；还是list的处理快啊！！！
            X_train.append(row[6])
            y_train.append(int(row[4]))
    return X_train, y_train


def split_words(text_line):
    lmtzr = nltk.WordNetLemmatizer()
    text_token = CountVectorizer().build_tokenizer()(text_line.lower())  #去掉标点，去单个词，小写化
    text_token = [lmtzr.lemmatize(w) for w in text_token]  #词干归一化

    return text_token


##高级操作，“生成器”做词语料库；不清楚为什么用函数写不行
class W2V_corpus_iter():
    def __init__(self, X_train):
        self.X_train = X_train

    def __iter__(self):
        for paragraph in self.X_train:
            for sent in paragraph:
                yield sent


def train_W2V(sentenList, file):
    w2vModel = Word2Vec(sentences=sentenList, iter=4, size=300, hs=0, negative=5, min_count=5, window=4)
    w2vModel.save(file)
    return w2vModel


def load_W2V(file, mode):
    if mode == 'rb':
        w2vModel = gensim.models.word2vec.Word2Vec.load(file)
    elif mode == 'rt':
        try:
            w2vModel = gensim.models.KeyedVectors.load_word2vec_format(file, binary=False)
        except:
            print('load by myself')
            f = open(file, encoding='utf-8')
            embedd_dim = len(next(f).rstrip().split()[1:])
            w2vModel = {}
            for line in f:
                line = line.rstrip().split()
                w = line[0]
                v = line[1:]
                if len(v) == embedd_dim:
                    w2vModel[w] = v
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


def make_X_train_idx_(X_train, word2idx, maxNum, maxLen):
    X_train_idx = [[[word2idx.get(w, 0) for w in sent] for sent in paragraph] for paragraph in X_train]

    for paragraph in X_train_idx:
        num_diff = maxNum - len(paragraph)
        if num_diff > 0:
            for _ in range(num_diff):
                paragraph.append([0] * maxLen)  #可变对象，针对元素的操作，还是改变自身，也即引用
        elif num_diff < 0:
            for _ in range(-num_diff):
                paragraph.pop()

    for i in range(len(X_train_idx)):
        X_train_idx[i] = sequence.pad_sequences(X_train_idx[i], maxlen=maxLen, padding='post')  #元素级的操作，还是引用
    return np.array(X_train_idx)


def make_X_train_idx(X_train, word2idx, maxNum, maxLen):
    X_train_idx = [[[word2idx.get(w, 0) for w in sent] for sent in paragraph] for paragraph in X_train]

    new_X_train_idx = []
    for paragraph in X_train_idx:
        num_diff = maxNum - len(paragraph)
        if num_diff > 0:
            paragraph += ([[0] * maxLen] * num_diff)
        else:
            paragraph = paragraph[0: maxNum]
        new_X_train_idx.append(paragraph)

    for i in range(len(new_X_train_idx)):
        new_X_train_idx[i] = sequence.pad_sequences(new_X_train_idx[i], maxlen=maxLen, padding='post')  #元素级的操作，还是引用
    return np.array(new_X_train_idx)


def make_y_train_oneHot(y_train_idx, is_cate_dict):
    if is_cate_dict:
        cate_dict = {1:0, 2:1, 3:2, 4:3, 5:4}
        for i in range(len(y_train_idx)):
            y_train_idx[i] = cate_dict[y_train_idx[i]]

    y_set = set(y_train_idx)
    if y_set != set(range(len(y_set))):
        raise KeyError

    y_train_oneHot = np.zeros([len(y_train_idx), len(set(y_train_idx))])
    for i in range(0, len(y_train_oneHot)):
        y_train_oneHot[i][y_train_idx[i]] = 1

    return np.array(y_train_oneHot)


def evaluate_matrix(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    print('Acc:', accuracy_score(y_true, y_pred))
    print('precision:', precision_score(y_true, y_pred, average='macro'))
    print('recall:', recall_score(y_true, y_pred, average='macro'))
    print('F1:', f1_score(y_true, y_pred, average='macro'))
    return f1_score(y_true, y_pred, average='macro')