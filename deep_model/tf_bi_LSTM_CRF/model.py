import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from data_utils import *
from time import time, localtime, strftime
tf.reset_default_graph()  # 重设整个“图”

class bi_lstm_crf_model():
    def __init__(self, args):
        #参数设定
        self.char2idx = args.char2idx
        self.idx2char = args.idx2char
        self.embedd_matrix = args.embedd_matrix
        self.hs = args.hidden_size
        self.num_tags = args.num_tags
        self.lr = args.lr
        self.bs = args.batch_size
        self.dropout = args.dropout
        self.n_epoch = args.n_epoch
        self.n_gs_to_display = args.n_gs_to_display
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.maxLen = args.maxLen
        self.idx2tag = args.idx2tag
        self.n_max_model = args.n_max_model
        self.n_gs_to_save_model = args.n_gs_to_save_model

        #建立模型！！
        self.built_model()


    def built_model(self):
        self.sentences = tf.placeholder(tf.int32, shape=[None, None], name='sentences')  #(bs, max_seq)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')  #(bs, max_seq)
        self.sequences_len = tf.placeholder(tf.int32, shape=[None], name='sequences_len') #(bs,)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')


        with tf.name_scope('embedd_layer'):
            embedd_matrix = tf.Variable(tf.cast(self.embedd_matrix, dtype='float32'),
                                        trainable=False, name='embedd_matrix')
            self.sentences_embedd = tf.nn.embedding_lookup(embedd_matrix, self.sentences)  # (bs, max_seq, embedd_size)


        with tf.name_scope('biLstm_layer'):
            cell_fw = LSTMCell(self.hs)
            cell_bw = LSTMCell(self.hs)
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                  cell_bw=cell_bw,
                                                                  inputs=self.sentences_embedd,
                                                                  sequence_length=self.sequences_len,
                                                                  dtype=tf.float32)
            lstm_out = tf.concat([out_fw, out_bw], axis=-1)
            self.lstm_out = tf.nn.dropout(lstm_out, self.dropout_keep_prob)  #(bs, max_seq, 2*hs)


        with tf.name_scope('hidden_layer'):
            W = tf.get_variable(name="W",
                                shape=[2*self.hs, self.num_tags],  #这个得注意：lstm每个cell的输出，都是共享W的
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(self.lstm_out)
            self.re_lstm_out = tf.reshape(self.lstm_out, [-1, 2*self.hs])  #(bs*max_seq, 2*hs)
            self.re_hidden_score = tf.nn.xw_plus_b(self.re_lstm_out, W, b)  #(bs*max_seq, num_tag)
            self.hidden_scores = tf.reshape(self.re_hidden_score, [-1, s[1], self.num_tags], name='hidden_score')
                                                                                            #(bs, max_seq, num_tag)

        with tf.name_scope('CRF_layer'):  #最关键的部分是这里，使用tf集成的库
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.hidden_scores,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequences_len)

        with tf.name_scope('loss_layer'):
            self.global_steps = tf.Variable(0, trainable=False, name='global_steps')
            self.loss = -tf.reduce_mean(log_likelihood, name='loss')
            self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss,
                                                                                global_step=self.global_steps)


class bi_lstm_crf_train(bi_lstm_crf_model):
    def fit(self, sentences_idx, labels_idx, sequences_len):
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.n_max_model)
            sess.run(tf.global_variables_initializer())  # 随机初始化变量
            tf.summary.FileWriter(self.model_path + self.model_name, sess.graph)  # 打开记录图

            ckpt = tf.train.get_checkpoint_state(self.model_path)  #这3行代码是为了实现断点续训！
            if ckpt and ckpt.model_checkpoint_path:  #如果里面保存了模型，就在那个模型的基础上训练；否则重新训练！！
                saver.restore(sess, ckpt.model_checkpoint_path)  # 恢复保存的神经网络结构，实现断点续训

            for i_epoch in range(self.n_epoch):
                batch_yield_iter = batch_yield(sentences_idx, labels_idx, sequences_len, self.bs, is_shuffle=True)  #训练时必须要先打乱数据的！！
                for (bat_sens_idx, bat_labs_idx, bat_seqs_len) in batch_yield_iter:
                    loss, gs = self._train_one_batch(sess, bat_sens_idx, bat_labs_idx, bat_seqs_len, self.dropout)

                    if gs % self.n_gs_to_display == 0:
                        time = strftime("%H:%M:%S", localtime())
                        print('{}>>epoch:{}>>gs:{}>>loss:{:.4}'.format(time, i_epoch, gs, loss))

                    if gs % self.n_gs_to_save_model == 0:
                        saver.save(sess, self.model_path + self.model_name + "_{}".format(gs))  #保存每个epoch的模型


    def predict(self, ckpt_model, sentences_idx, sequences_len):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_model)  #从加载“图”

            labels = [0] * len(sentences_idx)  #这个是没用的
            pred_labels = []
            data_iter = batch_yield(sentences_idx, labels, sequences_len, self.bs)
            for (bat_sens, _, bat_seqs_len) in data_iter:
                bat_pred_labs = self._predict_one_batch(sess, bat_sens, bat_seqs_len)
                pred_labels += [[self.idx2tag[idx] for idx in labs] for labs in bat_pred_labs]

        return pred_labels  #直接返回最终的结果


    def predict_one_sentence(self, ckpt_model, one_sen_str):
        one_sentence_list = [[char for char in one_sen_str]]
        one_sentence_list, seq_len = pad_sequences(one_sentence_list, self.maxLen)
        one_sentence_idx = sequences2idx(one_sentence_list, self.char2idx)  ##二维列表，注意
        one_label = self.predict(ckpt_model, one_sentence_idx, seq_len)[0]
        return one_label



    def _train_one_batch(self, sess, bat_sens, bat_labs, seqs_len, dropout):
        feed_dict = {self.sentences: bat_sens,
                     self.labels: bat_labs,
                     self.sequences_len: seqs_len,
                     self.dropout_keep_prob: dropout}
        _, loss, gs = sess.run([self.optim, self.loss, self.global_steps], feed_dict=feed_dict)
        return loss, gs


    def _predict_one_batch(self, sess, bat_sens, bat_seqs_len):
        feed_dict = {self.sentences: bat_sens,
                     self.sequences_len: bat_seqs_len,
                     self.dropout_keep_prob: 1.0}
        hidden_scores, transition_params = sess.run([self.hidden_scores, self.transition_params],
                                                    feed_dict=feed_dict)

        bat_labels = []
        for scocre, seq_len in zip(hidden_scores, bat_seqs_len):
            labs, _ = viterbi_decode(scocre[:seq_len], transition_params)
            bat_labels.append(list(labs))

        return bat_labels

