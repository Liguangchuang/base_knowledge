import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from data_utils import sequences2idx, pad_sequences, batch_yield, get_entitys, get_splitWord, \
    load_W2V, build_char2idx_embedMatrix



class seq_out():
    def __init__(self, mode='SW'):
        in_path = 'D:/python_code/AI_data/bi_lstm_crf/'
        dict_w2vModel = load_W2V(in_path + 'Tencent_char_10000.txt')
        self.char2idx, embedd_matrix = build_char2idx_embedMatrix(dict_w2vModel)

        if mode == 'NER':
            model_path = in_path + 'NER/'
            model_name = 'bi_lstm_crf_NER_4000'
            idx2tag = {0: 'O',
                       1: "B-PER", 2: "I-PER",
                       3: "B-LOC", 4: "I-LOC",
                       5: "B-ORG", 6: "I-ORG"}
            get_prediction = get_entitys
            maxLen = 150

        elif mode == 'SW':
            model_name = 'bi_lstm_crf_SW_5300'
            model_path = in_path + 'SW/'
            idx2tag = {0: 'S', 1: "B", 2: "I"}
            get_prediction = get_splitWord
            maxLen = 350

        self.model_path = model_path
        self.model_name = model_name
        self.idx2tag = idx2tag
        self.get_prediction = get_prediction
        self.maxLen = maxLen

        self.saver = tf.train.import_meta_graph(self.model_path + self.model_name + ".meta")
        graph = tf.get_default_graph()
        self.sentences = graph.get_tensor_by_name("sentences:0")
        self.sequences_len = graph.get_tensor_by_name("sequences_len:0")
        self.dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
        self.hidden_scores = graph.get_tensor_by_name("hidden_layer/hidden_score:0")
        self.transition_params = graph.get_tensor_by_name("transitions:0")  #这个变量找得你好苦啊！！！

    def _process_data(self, sentences):
        with open(sentences, 'rt', encoding='utf-8') as f:
            sentences = []
            for line in f:
                sentences.append(line.strip())
        return sentences


    def _out_file(self, result, out_file):
        with open(out_file, 'wt', encoding='utf-8') as f:
            for sen in result:
                sen = ' '.join(sen) + '\n'
                f.write(sen)


    def out(self, sentences, out_file=None):
        '''
        :param sentences: 支持两种输入格式，1种是输入txt文件，一种是输入list
        :return:
        '''
        sentences_list = [[char for char in sen] for sen in sentences]
        sentences_list, sequences_len = pad_sequences(sentences_list, self.maxLen)
        sentences_idx = sequences2idx(sentences_list, self.char2idx)
        sequences_len = [seq if seq <= self.maxLen else self.maxLen for seq in sequences_len]

        if type(out_file) == str:
            fw = open(out_file, 'wt', encoding='utf-8')

        with tf.Session() as sess:
            self.saver.restore(sess, self.model_path + self.model_name)

            labels = [0] * len(sentences_idx)  # 这个是没用的
            pred_labels = []
            for (bat_sens, _, bat_seqs_len) in batch_yield(sentences_idx, labels, sequences_len, bs=500):
                feed_dict = {self.sentences: bat_sens,
                             self.sequences_len: bat_seqs_len,
                             self.dropout_keep_prob: 1.0}
                hidden_scores, transition_params = sess.run([self.hidden_scores, self.transition_params],
                                                            feed_dict=feed_dict)
                bat_labels = []
                for scocre, seq_len in zip(hidden_scores, bat_seqs_len):
                    labs, _ = viterbi_decode(scocre[:seq_len], transition_params)
                    bat_labels.append(list(labs))

                pred_labels += [[self.idx2tag[idx] for idx in labs] for labs in bat_labels]

            result = []
            for one_lab, one_sen_str in zip(pred_labels, sentences):
                result.append(self.get_prediction(one_lab, one_sen_str))

            if type(out_file)==str:
                self._out_file(result, out_file)

            return result



