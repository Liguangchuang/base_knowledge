import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

MAX_SEQ_LEN = 80
in_path = 'D:/python_code/AI_data/'
model_path = in_path + 'IAN/'



class IAN_model():
    def __init__(self, embedd_matrix):
        # 网络参数
        self.bs = 128
        self.sqe_len = MAX_SEQ_LEN
        self.num_class = 3
        self.hidden_size = 300
        self.embedd_matrix = embedd_matrix
        self.vocab_size = len(embedd_matrix)
        self.embedd_size = len(embedd_matrix[0])

        self.display = 20
        self.n_epoch = 70
        self.model_path = model_path
        self.model_name = 'IAN_model'
        self.graph_name = 'IAN_model.graph'

        # 建立模型
        self.built_model()

        # 保存模型
        self.saver = tf.train.Saver(tf.global_variables())


    def built_model(self):
        self.input_X_c = tf.placeholder(tf.int32, [None, self.sqe_len], name='input_X_c')  # (bs, sqe_len)
        self.input_X_t = tf.placeholder(tf.int32, [None, self.sqe_len], name='input_X_t')  # (bs, sqe_len)
        self.input_y = tf.placeholder(tf.float32, [None, self.num_class], name='input_y')  # (bs, num_class)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')


        with tf.name_scope('embedd_layers'):
            embedd_matrix = tf.Variable(tf.cast(self.embedd_matrix, tf.float32),
                                        trainable=False, name='embedd_matrix')
            self.embedd_c = tf.nn.embedding_lookup(embedd_matrix, self.input_X_c)  # (bs, sqe_len, embedd_size)
            self.embedd_t = tf.nn.embedding_lookup(embedd_matrix, self.input_X_t)  # (bs, sqe_len, embedd_size)


        with tf.name_scope('lstm_layers'):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            init_state = cell.zero_state(self.bs, tf.float32)

            #lstm_c -> (bs, sqe_len, hs)
            self.lstm_c, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedd_c,initial_state=init_state)
            self.lstm_t, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedd_t, initial_state=init_state)


        with tf.name_scope('attention_layers'):
            self.atten_c = self._attention(self.lstm_c, self.lstm_t)
            self.atten_t = self._attention(self.lstm_t, self.lstm_c)


        with tf.name_scope('concate_layers'):
            self.concatenate = tf.concat([self.atten_c, self.atten_t], 1)


        with tf.name_scope('output_layers'):
            self.output = layers.fully_connected(inputs=self.concatenate, num_outputs=self.num_class,
                                                 activation_fn=None)  # (bs, num_class)
            self.predictions = tf.argmax(self.output, 1, name='predictions')  # (bs, )


        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
            self.optim = tf.train.AdamOptimizer(learning_rate=0.02).minimize(self.loss)


        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')


    def _attention(self, lstm_c, lstm_t):
        # lstm_c -> (bs, seq_len, hs); lstm_c -> (bs, seq_len, hs)

        W_c = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size], stddev=0.1), name='W_c')
        b_c = tf.Variable(tf.constant(0.1, shape=[self.sqe_len, 1]), name='b')

        avg_t = tf.reduce_mean(lstm_t, axis=1)  # (bs, hs)
        avg_t = tf.reduce_mean(avg_t, axis=0, keep_dims=True)  # (1, hs)
        avg_t = tf.transpose(avg_t)  #(hs, 1)

        lstm_c_reshape = tf.reshape(lstm_c, shape=[-1, self.hidden_size])  # (bs * seq_len, hs)

        tmp_c_reshape = tf.matmul(lstm_c_reshape, W_c)  # (bs * sqe_len, hs)
        tmp_c_reshape = tf.matmul(tmp_c_reshape, avg_t)   # (bs * sqe_len, 1)
        tmp_c = tf.reshape(tmp_c_reshape, shape=[-1, self.sqe_len, 1])  # (bs, sqe_len, 1)
        tmp_c = tmp_c + b_c  # (bs, sqe_len, 1)
        att_score_c = tf.nn.tanh(tmp_c, name='att_score')  # (bs, sqe_len, 1)

        alpha_c = tf.nn.softmax(att_score_c, dim=1)  # (bs, seq_len, 1)
        atten_c = tf.reduce_sum(tf.multiply(lstm_c, alpha_c), axis=1)  # (bs, hs)

        return atten_c



class IAN_train_test(IAN_model):
    def train(self, data_list):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())  # 随机初始化变量
        tf.summary.FileWriter(self.model_path + self.graph_name, sess.graph)  # 打开记录图

        best_acc = 0
        for i_epoch in range(self.n_epoch):
            print('epech:', i_epoch, '>' * 50)

            X_tra_c = data_list[0]
            X_tra_t = data_list[1]
            y_tra = data_list[2]
            for bs_X_c, bs_X_t, bs_y, i_bat in self._batch_iter(X_tra_c, X_tra_t, y_tra, self.bs):
                feed_dict_tra = {self.input_X_c: bs_X_c,
                                 self.input_X_t: bs_X_t,
                                 self.input_y: bs_y,
                                 self.dropout_keep_prob: 0.2}
                _, loss, tra_acc = sess.run([self.optim, self.loss, self.accuracy], feed_dict=feed_dict_tra)

                if len(data_list) == 3:
                    if tra_acc > best_acc:
                        best_acc = tra_acc
                        self.saver.save(sess, self.model_path + self.model_name)
                        print('best_acc:{} >>> model had save'.format(best_acc))
                    if i_bat % self.display == 0:
                        print('loss:{} >>> train_acc:{}'.format(loss, tra_acc))

                elif len(data_list) == 6:
                    X_val_c = data_list[3]
                    X_val_t = data_list[4]
                    y_val = data_list[5]
                    feed_dict_val = {self.input_X_c: X_val_c,
                                     self.input_X_t: X_val_t,
                                     self.input_y: y_val,
                                     self.dropout_keep_prob: 0.2}
                    val_acc = sess.run(self.accuracy, feed_dict=feed_dict_val)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        self.saver.save(sess, self.model_path + self.model_name)
                        print('best_acc:{} >>> model had save'.format(best_acc))

                    if i_bat % self.display == 0:
                        print('loss:{} >>> train_acc:{} >>> val_acc{}'.format(loss, tra_acc, val_acc))

    def test(self, test_data):
        X_test_c = test_data[0]
        X_test_t = test_data[1]
        y_test = test_data[2]

        graph = tf.get_default_graph()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

        input_X_c = graph.get_tensor_by_name("input_X_c:0")
        input_X_t = graph.get_tensor_by_name("input_X_t:0")
        input_y = graph.get_tensor_by_name("input_y:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")

        feed_dict_test = {input_X_c: X_test_c,
                          input_X_t: X_test_t,
                          input_y: y_test,
                          dropout_keep_prob: 1.0}

        accuracy = graph.get_tensor_by_name("accuracy/accuracy:0")
        test_acc = sess.run(accuracy, feed_dict=feed_dict_test)
        print('test_acc:', test_acc)

        predictions = graph.get_tensor_by_name("output_layers/predictions:0")
        y_pred = sess.run(predictions, feed_dict=feed_dict_test)

        return y_pred


    def _batch_iter(self, X_c, X_t, y, bs):
        data_len = len(X_c)
        n_batch = int(data_len / bs)  #很奇怪，rnn，必须要一样的bs

        idx_shuffle = np.random.permutation(np.arange(data_len))
        X_shuffle_c = np.array([X_c[i] for i in idx_shuffle])
        X_shuffle_t = np.array([X_t[i] for i in idx_shuffle])
        y_shuffle = np.array([y[i] for i in idx_shuffle])

        for i_bat in range(n_batch):
            sta_idx = i_bat * bs
            end_idx = min((i_bat + 1) * bs, data_len)
            yield X_shuffle_c[sta_idx: end_idx], X_shuffle_t[sta_idx: end_idx], y_shuffle[sta_idx: end_idx], i_bat