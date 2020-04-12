import numpy as np
import tensorflow as tf
import os
from time import time

MAX_SEQ_LEN = 250
in_path = 'D:/python_code/AI_data/'

class textCNN_model():
    def __init__(self, embedMatrix):
        # 网络参数
        self.sqe_len = MAX_SEQ_LEN
        self.num_class = 2
        self.embedd_matrix = embedMatrix
        self.vocab_size = len(embedMatrix)
        self.embedd_size = len(embedMatrix[0])
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 30
        self.lr = 0.02
        self.global_steps = tf.Variable(0, trainable=False)

        #训练呢参数
        self.n_display = 1
        self.n_epoch = 1
        self.bs = 500
        self.model_path = in_path + 'model/textCNN_model/'
        self.model_name = 'textCNN_model'
        self.graph_name = 'textCNN_graph'

        # 建立模型
        self.built_model()

        #保存模型
        self.saver = tf.train.Saver(tf.global_variables())

    def built_model(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sqe_len], name='input_x')  # (bs, sqe_len)
        self.input_y = tf.placeholder(tf.float32, [None, self.num_class], name='input_y')  # (bs, num_class)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')


        with tf.name_scope('embedding_layers'):
            W = tf.Variable(tf.cast(self.embedd_matrix, dtype='float32'),trainable=False, name='W')  # 直接用常量初始化embedd_matrix
            self.embedd_input_x = tf.nn.embedding_lookup(W, self.input_x)  # (bs, seq_len, embedd_size)
            self.embedd_input_x_expanded = tf.expand_dims(self.embedd_input_x, -1)  # (bs, seq_len, embedd_size, 1)


        pool_outputs = []
        for i, fsz in enumerate(self.filter_sizes):
            with tf.name_scope('conv_maxpool_{}_layers'.format(fsz)):
                filter_shape = [fsz, self.embedd_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters], name='b'))
                conv = tf.nn.conv2d(self.embedd_input_x_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')  # (bs, sqe_len-filter_sizes+1, 1, num_filters)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')  # (bs, sqe_len-filter_sizes+1, 1, num_filters)
                pool = tf.nn.max_pool(h,
                                      ksize=[1, self.sqe_len - fsz + 1, 1, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='VALID',
                                      name='pool')  # (bs, 1, 1, num_filters)
                pool_outputs.append(pool)
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pool_outputs, 3)  # (bs, 1, 1, num_filters_total)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])  # (bs, num_filters_total)


        with tf.name_scope('dropout_layers'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)  # (bs, num_filters_total)


        with tf.name_scope('full_layers'):
            W = tf.Variable(tf.truncated_normal([num_filters_total, self.num_class], stddev=0.1), name='W')
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


class textCNN_train_test(textCNN_model):
    def train(self, data_list, is_save=True):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 随机初始化变量
            tf.summary.FileWriter(self.model_path + self.graph_name, sess.graph)  # 打开记录图

            X_tra = data_list[0]
            y_tra = data_list[1]
            if len(data_list) == 4:
                X_val = data_list[2]
                y_val = data_list[3]

            best_acc = 0
            for i_epoch in range(self.n_epoch):
                print('epoch:', i_epoch, '>' * 50)
                for bs_X, bs_y, _ in self._shuffle_batch_iter(X_tra, y_tra, self.bs, is_shuffle=True):
                    loss, tra_acc, gs = self._train_step(sess, bs_X, bs_y)


                    if len(data_list) == 2:
                        if gs % self.n_display == 0:
                            print('gs:{} >>> loss:{:.4} >>> train_acc:{:.4}'.format(gs, loss, tra_acc))
                            if (tra_acc > best_acc) and is_save:
                                best_acc = tra_acc
                                self.saver.save(sess, self.model_path + self.model_name)
                                print('best_acc:{:.4} >>> model had save'.format(best_acc))

                    elif len(data_list) == 4:
                        if gs % self.n_display == 0:
                            val_acc = self._val_step(sess, X_val, y_val)
                            print('gs:{} >>> loss:{:.4} >>> train_acc:{:.4} >>> val_acc{:.4}'.format(gs, loss, tra_acc, val_acc))
                            if (val_acc > best_acc) and is_save:
                                best_acc = val_acc
                                self.saver.save(sess, self.model_path + self.model_name)
                                print('best_acc:{:.4} >>> model had save'.format(best_acc))


    def test(self, test_data):
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)

            X_test = test_data[0]
            y_test = test_data[1]

            y_pred = []   #写得这么折腾，是因为batch过大，会报内存错误
            for bs_X, bs_y, _ in self._shuffle_batch_iter(X_test, y_test, self.bs):
                feed_dict = {self.input_x: bs_X,
                             self.dropout_keep_prob: 1.0}
                bat_y_pred = sess.run(self.predictions, feed_dict=feed_dict)

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


    def _train_step(self, sess, bs_X, bs_y, dropout=0.2):
        feed_dict = {self.input_x: bs_X,
                     self.input_y: bs_y,
                     self.dropout_keep_prob: dropout}
        _, loss, tra_acc, gs = sess.run([self.optim, self.loss, self.accuracy, self.global_steps], feed_dict=feed_dict)

        return loss, tra_acc, gs


    def _val_step(self, sess, X_val, y_val):
        n_corr = 0
        for bs_X, bs_y, _ in self._shuffle_batch_iter(X_val, y_val, self.bs):
            feed_dict_val = {self.input_x: bs_X,
                             self.input_y: bs_y,
                             self.dropout_keep_prob: 1}
            bat_n_corr = sess.run(self.n_correct, feed_dict=feed_dict_val)
            n_corr += bat_n_corr
        val_acc = n_corr / len(X_val)

        return val_acc

