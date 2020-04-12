import numpy as np
import tensorflow as tf
import argparse
from data_utils import (load_W2V, build_char2idx_embedMatrix, read_corpus, pad_sequences, get_entitys,
                        sequences2idx, evaluate, get_splitWord, seq_data_process, get_pred_labels)
from model import biLstm_crf_model
from time import time
import jieba
from keras.callbacks import ModelCheckpoint

#######################################################################
# 这部分代码是，设置各种参数！！！！

parser = argparse.ArgumentParser(description='BiLSTM-CRF')
parser.add_argument('--mode', type=str, default='demo')
parser.add_argument('--sence', type=str, default='NER')
out_args = parser.parse_args()


class args():
    def __init__(self):
        in_path = 'D:/python_code/AI_data/bi_lstm_crf/'

        dict_w2vModel = load_W2V(in_path + 'Tencent_char_10000.txt')
        char2idx, embedd_matrix = build_char2idx_embedMatrix(dict_w2vModel)
        idx2char = {idx: char for (char, idx) in char2idx.items()}

        if out_args.sence == 'NER':
            print('NER')
            tag2idx = {"O": 0,
                       "B-PER": 1, "I-PER": 2,
                       "B-LOC": 3, "I-LOC": 4,
                       "B-ORG": 5, "I-ORG": 6}
            idx2tag = {0: 'O',
                       1: "B-PER", 2: "I-PER",
                       3: "B-LOC", 4: "I-LOC",
                       5: "B-ORG", 6: "I-ORG"}
            model_path = in_path + 'NER/'
            model_name = 'bi_lstm_crf_NER'
            train_file_name = in_path + 'train_data_NER'
            test_file_name = in_path + 'test_data_NER'
            maxLen = 150
            get_prediction = get_entitys
            demo_ckpt_model = model_path + model_name + '.h5'

        else:
            print('SW')
            tag2idx = {"S": 0, "B": 1, "I": 2}
            idx2tag = {0: 'S', 1: "B", 2: "I"}
            model_path = in_path + 'SW/'
            model_name = 'bi_lstm_crf_SW'
            train_file_name = in_path + 'train_data_SW'
            test_file_name = in_path + 'test_data_SW'
            maxLen = 350
            get_prediction = get_splitWord
            demo_ckpt_model = model_path + model_name + '.h5'

        self.in_path = in_path
        self.model_path = model_path
        self.model_name = model_name

        self.embedd_matrix = embedd_matrix
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.num_tags = len(tag2idx)
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.get_prediction = get_prediction
        self.demo_ckpt_model = demo_ckpt_model
        self.maxLen = maxLen

        self.hidden_size = 300
        self.lr = 0.001
        self.batch_size = 500
        self.dropout = 0.5
        self.n_epoch = 50
        self.n_gs_to_display = 50
        self.n_gs_to_save_model = 100
        self.n_max_model = 50


args = args()



if __name__ == '__main__':
    if out_args.mode == 'train' or out_args.mode == 'test':
        print('——————load_data——————')
        train_sentences, train_labels = read_corpus(args.train_file_name)
        train_sentences_idx, train_labels_idx, train_sentences_len = seq_data_process(train_sentences, train_labels,
                                                                                      args)
        print('train_data_len:', len(train_sentences_idx))

        test_sentences, test_labels = read_corpus(args.test_file_name)
        test_sentences_idx, test_labels_idx, test_sentences_len = seq_data_process(test_sentences, test_labels, args)
        print('test_data_len:', len(test_sentences_idx))

        # keras的特殊数据处理，与tf有点不一样的地方！！
        train_labels_idx = [[[idx] for idx in lab] for lab in train_labels_idx]
        train_sentences_idx = np.array(train_sentences_idx)
        train_labels_idx = np.array(train_labels_idx)  # 很蛋疼，keras一定要np.array的类型，不能是原生的list

        test_labels_idx = [[[idx] for idx in lab] for lab in test_labels_idx]
        test_sentences_idx = np.array(test_sentences_idx)
        test_labels_idx = np.array(test_labels_idx)  # 很蛋疼，keras一定要np.array的类型，不能是原生的list



        if out_args.mode == 'train':
            print('——————train——————')
            model = biLstm_crf_model(args)
            model.fit(train_sentences_idx, train_labels_idx, epochs=2, batch_size=500, verbose=1)
            model.save(args.model_path + args.model_name + '.h5')  # 好像一定要有.h5作为结尾，真奇怪！

        elif out_args.mode == 'test':
            print('——————test——————')
            ###############################################
            model = biLstm_crf_model(args)
            model.load_weights(args.model_path + args.model_name + '.h5')
            predictions = model.predict(test_sentences_idx)
            pred_labels = get_pred_labels(predictions, test_sentences_len, args.idx2tag)
            f1 = evaluate(test_labels, pred_labels)
            print(f1)
            print(test_sentences[10: 15])
            print('test:', test_labels[10: 15])
            print('pred:', pred_labels[10: 15])


    elif out_args.mode == 'demo':
        print('————————————demo————————————')
        model = biLstm_crf_model(args)
        model.load_weights(args.model_path + args.model_name + '.h5')
        while True:
            print('pleace input a sentence:')
            one_sentence_str = input()
            if one_sentence_str != '':
                one_sentence_list = [[char for char in one_sentence_str]]
                one_sentence_list, seq_len = pad_sequences(one_sentence_list, args.maxLen)
                one_sentence_idx = sequences2idx(one_sentence_list, args.char2idx)  ##二维列表，注意

                predictions = model.predict(np.array(one_sentence_idx))  #这个是不是有点问题啊？转移矩阵去哪里啦？
                one_label = get_pred_labels(predictions, seq_len, args.idx2tag)[0]
                print(one_label)

                prediction_list = args.get_prediction(one_label, one_sentence_str)
                print(prediction_list, '\n')

