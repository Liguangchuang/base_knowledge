import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import os
from time import time

tf.reset_default_graph()  # 重设整个“图”
MAX_SEQ_LEN = 250
in_path = 'D:/python_code/AI_data/'

class LSTM_model():
    def __init__(self, embedMatrix):
        # 网络参数
        self.sqe_len = MAX_SEQ_LEN
        self.num_class = 2
        self.embedd_matrix = embedMatrix
        self.vocab_size = len(embedMatrix)
        self.embedd_size = len(embedMatrix[0])
        self.hidden_size = 300
        self.lr = 0.02
        self.global_steps = tf.Variable(0, trainable=False)

        #训练呢参数
        self.n_save = 1
        self.n_display = 1
        self.n_epoch = 1
        self.bs = 50
        self.model_path = in_path + 'model/'
        self.model_name = 'LSTM_model'
        self.graph_name = 'LSTM.graph'

        # 建立模型
        self.built_model()

        #保存模型
        self.saver = tf.train.Saver(tf.global_variables())

    def built_model(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sqe_len], name='input_x')  # (bs, sqe_len)
        self.input_y = tf.placeholder(tf.float32, [None, self.num_class], name='input_y')  # (bs, num_class)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')


        with tf.device('/cpu:0'), tf.name_scope('embedding_layers'):
            W = tf.Variable(tf.cast(self.embedd_matrix, dtype='float32'),
                            trainable=False, name='W')  # 直接用常量初始化embedd_matrix
            self.embedd_input_x = tf.nn.embedding_lookup(W, self.input_x)  # (bs, seq_len, embedd_size)
            print(self.embedd_input_x.shape)


        with tf.name_scope('LSTM_layers'):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            init_state = cell.zero_state(self.bs, tf.float32)

            #lstm_outputs -> (bs, sqe_len, hs)
            self.lstm_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedd_input_x,
                                                               initial_state=init_state)
            print(self.lstm_outputs.shape)   #奇怪，忽然bs变成指定长度，因此我设置的self.bs=49会报错


        with tf.name_scope('dropout_layers'):
            self.h_drop = tf.nn.dropout(self.lstm_outputs[:, -1, :], self.dropout_keep_prob)  # (bs, hs)
            print(self.h_drop.shape)


        with tf.name_scope('full_layers'):
            W = tf.Variable(tf.truncated_normal([self.hidden_size, self.num_class], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.num_class]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='score')  # (bs, num_class)
            self.predictions = tf.argmax(self.scores, 1, name='predictions')  # (bs, )


        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss,
                                                                                global_step=self.global_steps)

        # calculate accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.cast(tf.equal(self.predictions, tf.argmax(self.input_y, 1)), 'float')
            self.n_correct = tf.reduce_sum(correct_predictions, name='accuracy')
            self.accuracy = tf.reduce_mean(correct_predictions, name='accuracy')


class LSTM_train_test(LSTM_model):
    def train(self, data_list, is_save=True):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())  # 随机初始化变量
        tf.summary.FileWriter(self.model_path + self.graph_name, sess.graph)  # 打开记录图

        best_acc = 0
        for i_epoch in range(self.n_epoch):
            print('epech:', i_epoch, '>' * 50)

            X_tra = data_list[0]
            y_tra = data_list[1]
            for bs_X, bs_y, i_bat in self._shuffle_batch_iter(X_tra, y_tra, self.bs, is_shuffle=True):
                print(len(bs_X))
                feed_dict_tra = {self.input_x: bs_X,
                                 self.input_y: bs_y,
                                 self.dropout_keep_prob: 0.2}
                _, loss, tra_acc, gs = sess.run([self.optim, self.loss, self.accuracy, self.global_steps],
                                                feed_dict=feed_dict_tra)

                if gs % self.n_display == 0:
                    if len(data_list) == 2:
                        print('loss:{} >>> train_acc:{}'.format(loss, tra_acc))
                        if (tra_acc > best_acc) and is_save:
                            best_acc = tra_acc
                            self.saver.save(sess, self.model_path + self.model_name)
                            print('best_acc:{} >>> model had save'.format(best_acc))


                    elif len(data_list) == 4:
                        X_val = data_list[2]
                        y_val = data_list[3]
                        n_corr = 0
                        for bs_X, bs_y, _ in self._shuffle_batch_iter(X_val, y_val, self.bs):
                            feed_dict_val = {self.input_x: bs_X,
                                             self.input_y: bs_y,
                                             self.dropout_keep_prob: 1}
                            bat_n_corr = sess.run(self.n_correct, feed_dict=feed_dict_val)
                            n_corr += bat_n_corr
                        val_acc = n_corr / len(X_val)

                        print('loss:{} >>> train_acc:{} >>> val_acc{}'.format(loss, tra_acc, val_acc))
                        if (val_acc > best_acc) and is_save:
                            best_acc = val_acc
                            self.saver.save(sess, self.model_path + self.model_name)
                            print('best_acc:{} >>> model had save'.format(best_acc))



    def test(self, test_data):
        X_test = test_data[0]
        y_test = test_data[1]

        graph = tf.get_default_graph()  #这4行代码通用的，copy过去便可
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

        input_x = graph.get_tensor_by_name("input_x:0")
        input_y = graph.get_tensor_by_name("input_y:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
        predictions = graph.get_tensor_by_name("full_layers/predictions:0")

        y_pred = []   #写得这么折腾，是因为batch过大，会报内存错误
        for bs_X, bs_y, i_bat in self._shuffle_batch_iter(X_test, y_test, self.bs):
            feed_dict = {input_x: bs_X,
                         dropout_keep_prob: 1.0}
            bat_y_pred = sess.run(predictions, feed_dict=feed_dict)

            y_pred += list(bat_y_pred)

        return np.array(y_pred)


    def _shuffle_batch_iter(self, X, y, bs, is_shuffle=False):
        data_len = len(X)
        n_batch = int(data_len / bs) if data_len % bs == 0 else int(data_len / bs) + 1

        if is_shuffle:
            idx_shuffle = np.random.permutation(np.arange(data_len))
            X_shuffle = np.array([X[i] for i in idx_shuffle])
            y_shuffle = np.array([y[i] for i in idx_shuffle])
        else:
            X_shuffle = X
            y_shuffle = y

        for i_bat in range(n_batch):
            sta_idx = i_bat * bs
            end_idx = min((i_bat + 1) * bs, data_len)
            yield X_shuffle[sta_idx: end_idx], y_shuffle[sta_idx: end_idx], i_bat