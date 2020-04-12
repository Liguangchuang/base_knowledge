import numpy as np
from sklearn.metrics import (r2_score, mean_absolute_error, confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)


def load_W2V(file, word_set=None):
    '''
    用于读取W2V，用一个字典保存
    :param file: 文件位置 + 文件名
    :param word_set:
    :return: 字典保存的模型，键：单词；值：向量
    '''
    with open(file, encoding='utf-8') as f:
        dict_w2vModel = {}
        first_line = next(f).rstrip().split()
        embedd_dim = len(first_line[1:])  # 第一行的向量长度
        dict_w2vModel[first_line[0]] = [float(e) for e in first_line[1:]]
        for line in f:
            line = line.rstrip().split()
            w = line[0]
            v = [float(e) for e in line[1:]]

            if (len(v) == embedd_dim):
                if word_set == None:
                    dict_w2vModel[w] = v
                else:
                    if w in word_set:
                        dict_w2vModel[w] = v  # 只选择出现的词，避免内存溢出

    return dict_w2vModel


def build_char2idx_embedMatrix(dict_w2vModel):
    '''
    :param dict_w2vModel: 用字典保存的模型，再处理成embedMatrix
    :return:char2idx 和 embedMatrix
    '''
    char2idx = {"_stopWord": 0}  # 这里加了一行是用来过滤停用词的。
    vocab_list = [(w, v) for w, v in dict_w2vModel.items()]
    embedMatrix = np.zeros((len(dict_w2vModel) + 1, len(vocab_list[0][1])))

    for i in range(0, len(vocab_list)):
        word = vocab_list[i][0]
        char2idx[word] = i + 1
        embedMatrix[i + 1] = vocab_list[i][1]

    return char2idx, embedMatrix


def read_corpus(file):
    '''
    用于读取标注的数据（通常的，以列排放的数据）
    :param file: 文件位置 + 文件名
    :return: 以列表的形式返回句子和标注
    '''
    with open(file, encoding='utf-8') as f:
        sentences, labels = [], []
        sen, lab = [], []
        for line in f:
            if line != '\n':
                line = line.strip()
                sen.append(line[0])
                lab.append(line[2:])
            else:
                sentences.append(sen)
                labels.append(lab)
                sen, lab = [], []

    return sentences, labels


def pad_sequences(sequences, maxLen, pad_mark=0):
    '''
    用于序列的补齐
    :param sequences:
    :param maxLen:
    :param pad_mark: 默认用0作为补齐标志
    :return: 补齐的句子，以及句子的原来长度
    '''
    pad_sequences_list, sequences_len = [], []
    for seq in sequences:
        sequences_len.append(len(seq))
        if len(seq) < maxLen:
            pad_sequences_list.append(seq + [pad_mark] * (maxLen - len(seq)))
        else:
            pad_sequences_list.append(seq[0: maxLen])

    return pad_sequences_list, sequences_len


def sequences2idx(sequences, char2idx):
    '''
    将序列中的字符，用idx表示
    :param sequences: 输入的序列
    :param char2idx: 字符->idx的映射字典。
    :return: 映射后的序列
    '''
    sequences_idx = [[char2idx.get(char, 0) for char in seq] for seq in sequences]
    return sequences_idx


def seq_data_process(train_sentences, train_labels, args):
    train_sentences_pad, train_sentences_len = pad_sequences(train_sentences, args.maxLen)
    train_labels_pad, _ = pad_sequences(train_labels, args.maxLen)
    train_sentences_idx = sequences2idx(train_sentences_pad, args.char2idx)
    train_labels_idx = sequences2idx(train_labels_pad, args.tag2idx)
    return train_sentences_idx, train_labels_idx, train_sentences_len


def batch_yield(X, y, seqs_len, bs, is_shuffle=False):
    '''
    用生成器，每次返回一个bs的数据；包括：bs个句子，bs个标签，每条句子的长度
    :param X: 全部的句子
    :param y: 全部的标签
    :param seqs_len: 全部数据中，每条句子的长度
    :param bs: 指定bs的大小
    :param is_shuffle: 是否打乱，（训练时要打乱，预测时不能打乱）
    :return:
    '''
    data_len = len(X)
    n_batch = int(data_len / bs) if data_len % bs == 0 else int(data_len / bs) + 1

    if is_shuffle:
        idx_shuffle = np.random.permutation(range(data_len))  #将序列打乱！！！
        X = [X[i] for i in idx_shuffle]
        y = [y[i] for i in idx_shuffle]
        seqs_len = [seqs_len[i] for i in idx_shuffle]

    for i_bat in range(n_batch):
        sta_idx = i_bat * bs
        end_idx = min((i_bat + 1) * bs, data_len)

        if (end_idx - sta_idx) != bs:
            print('specified bs is {}, but this bs is {}:'.format(bs, end_idx-sta_idx))

        yield X[sta_idx: end_idx], y[sta_idx: end_idx], seqs_len[sta_idx: end_idx]


def get_entitys(one_label, one_sentence_str):
    '''
    在标签序列中，提取实体
    :param one_label: 预测的标签
    :param one_sentence_str: 输入的句子
    :return:
    '''
    entity_list = []
    i = 0
    while i < len(one_label):
        if one_label[i] == "B-PER":  #这里只会收集以"B"为开头的实体，如果开头就预测为"I"，就是预测错，不用管
            start_idx = i
            i += 1
            while (i < len(one_label)) and (one_label[i] == "I-PER"):
                i += 1
            entity_list.append(('PER', one_sentence_str[start_idx: i]))

        elif one_label[i] == "B-LOC":
            start_idx = i
            i += 1
            while (i < len(one_label)) and (one_label[i] == "I-LOC"):
                i += 1
            entity_list.append(('LOC', one_sentence_str[start_idx: i]))

        elif one_label[i] == "B-ORG":
            start_idx = i
            i += 1
            while (i < len(one_label)) and (one_label[i] == "I-ORG"):
                i += 1
            entity_list.append(('ORG', one_sentence_str[start_idx: i]))

        else:
            i += 1

    return entity_list


def get_splitWord(one_label, one_sentence_str):
    '''
    用于将标注数据，转化为“分好词的句子”
    :param one_label: 预测的标签
    :param one_sentence_str: 输入的句子
    :return:
    '''
    splitWord_list = []

    i = 0
    while i < len(one_label):
        if one_label[i] == 'S':
            splitWord_list.append(one_sentence_str[i])
            i += 1
        elif one_label[i] == 'B':
            start_idx = i
            i += 1
            while (i < len(one_label)) and (one_label[i] == 'I'):
                i += 1
            splitWord_list.append(one_sentence_str[start_idx: i])
        else:  #如果开头是"I"就麻烦了，直接跳过了！
            i += 1

    return splitWord_list


def evaluate(true_labels, pred_labels):
    y_true = [tag for lab in true_labels for tag in lab]
    y_pred = [tag for lab in pred_labels for tag in lab]
    f1 = f1_score(y_true, y_pred, average='macro')

    return f1


def get_one_pred_lab(pred, seq_len, idx2tag):
    pred = pred[0: seq_len]
    lab_idx = [np.argmax(row) for row in pred]
    lab = [idx2tag[idx] for idx in lab_idx]

    return lab


def get_pred_labels(predictions, sequences_len, idx2tag):
    pred_labels = []
    for pred, seq_len in zip(predictions, sequences_len):
        lab = get_one_pred_lab(pred, seq_len, idx2tag)
        pred_labels.append(lab)

    return pred_labels




