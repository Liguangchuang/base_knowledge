import numpy as np
import tensorflow as tf
import argparse
from data_utils import (load_W2V, build_char2idx_embedMatrix, read_corpus, pad_sequences, get_entitys,
                        sequences2idx, evaluate, get_splitWord, seq_data_process)
from model import bi_lstm_crf_train
from time import time
import jieba


#######################################################################
#这部分代码是，设置各种参数！！！！

parser = argparse.ArgumentParser(description='BiLSTM-CRF')
parser.add_argument('--mode', type=str, default='demo')
parser.add_argument('--sence', type=str, default='SW')
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
            demo_ckpt_model = model_path + model_name + '_4400'
        
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
            demo_ckpt_model = model_path + model_name + '_5300'

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


def get_word_set(train_sentences, train_labels):
    train_word_set = set()
    for lab, sen in zip(train_labels, train_sentences):
        sen = ''.join(sen)
        word_list = args.get_prediction(lab, sen)
        if out_args.sence == 'NER':
            word_list = [tumple[1] for tumple in word_list]
        train_word_set |= set(word_list)
    return train_word_set


if __name__ == '__main__':
    if out_args.mode == 'train' or out_args.mode == 'test':
        print('——————load_data——————')
        start = time()
        train_sentences, train_labels = read_corpus(args.train_file_name)
        train_sentences_idx, train_labels_idx, train_sentences_len = seq_data_process(train_sentences,train_labels, args)
        print('train_data_len:', len(train_sentences_idx))

        test_sentences, test_labels = read_corpus(args.test_file_name)
        test_sentences_idx, test_labels_idx, test_sentences_len = seq_data_process(test_sentences, test_labels, args)
        print('test_data_len:', len(test_sentences_idx))

        train_word_set = get_word_set(train_sentences, train_labels)
        test_word_set = get_word_set(test_sentences, test_labels)
        new_word_set = test_word_set - (train_word_set & test_word_set)
        print('new_words:', len(new_word_set), new_word_set)
        print(time() - start)

        if out_args.mode == 'train':
            print('——————train——————')
            model = bi_lstm_crf_train(args)
            model.fit(train_sentences_idx, train_labels_idx, train_sentences_len)

        elif out_args.mode == 'test':
            print('——————test——————')
            ###############################################
            ckpt = tf.train.get_checkpoint_state(args.model_path)  # 所有模型的路径
            ckpt_model = ckpt.model_checkpoint_path  # 最后一个模型的路径

            model = bi_lstm_crf_train(args)
            pred_labels = model.predict(ckpt_model, test_sentences_idx, test_sentences_len)

            f1 = evaluate(test_labels, pred_labels)
            print(f1)
            print(test_sentences[10: 15])
            print('test:', test_labels[10: 15])
            print('pred:', pred_labels[10: 15])


            display_all_data =False
            if display_all_data:
                result = []
                for pred_lab, sen, test_lab in zip(pred_labels, test_sentences, test_labels):
                    sen = ''.join(sen)

                    word_list = args.get_prediction(test_lab, sen)
                    if out_args.sence == 'NER':
                        word_list = [tumple[1] for tumple in word_list]
                    line_new_word_set = set(word_list) & new_word_set
                    if line_new_word_set:
                        print(line_new_word_set)  #新词；看看新词的预测效果如何

                    print(args.get_prediction(pred_lab, sen))  #本模型的预测
                    print(jieba.lcut(sen))  #jieba预测
                    print('———————————————————————')

            ######################################################
            ##选择最佳的模型；避免过拟合！！！
            select_model = True
            if select_model:
                for model_idx in range(35, 54, 1):
                    model_idx = '_' + str(model_idx * args.n_gs_to_save_model)
                    ckpt_model = args.model_path + args.model_name + model_idx
                    pred_labels = model.predict(ckpt_model, test_sentences_idx, test_sentences_len)
                    f1 = evaluate(test_labels, pred_labels)
                    print((model_idx, f1))


    elif out_args.mode == 'demo':
            print('————————————demo————————————')
            model = bi_lstm_crf_train(args)
            while True:
                print('pleace input a sentence:')
                one_sentence_str = input()
                if one_sentence_str != '':
                    one_label = model.predict_one_sentence(args.demo_ckpt_model, one_sentence_str)
                    print(one_label)

                    prediction_list = args.get_prediction(one_label, one_sentence_str)
                    print(prediction_list, '\n')
