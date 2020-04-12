import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import numpy as np


MAX_SENT_NUM = 20
MAX_SENT_LEN = 60
NUM_CLASS = 5
in_path = 'D:/python_code/AI_data/'
model_path = in_path + 'HAN/'

class HAN_model():
    def __init__(self, embedd_matrix):
        # 网络参数
        self.num_class = NUM_CLASS
        self.max_sent_num = MAX_SENT_NUM
        self.max_sent_len = MAX_SENT_LEN
        self.hidden_size = 300
        self.embedd_matrix = embedd_matrix
        self.vocab_size = len(embedd_matrix)
        self.embedd_size = len(embedd_matrix[0])
        
        self.display = 1
        self.n_epoch = 20
        self.batch_size = 500
        self.model_path = model_path
        self.model_name = 'HAN_model'
        self.graph_name = 'HAN_graph'

        # 建立模型
        self.bulit_model()
        self.saver = tf.train.Saver(tf.global_variables())


    def bulit_model(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.max_sent_num,   # (bs, max_sent_num, max_sent_len)
                                                 self.max_sent_len], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_class], name='input_y')  # (bs, num_class)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')


        with tf.name_scope('word_embedd_layers'):
            embedd_matrix = tf.Variable(tf.cast(self.embedd_matrix, tf.float32), trainable=False, name='embedd_matrix')
            self.word_embedd = tf.nn.embedding_lookup(embedd_matrix,
                                                      self.input_x)  # (bs, max_sent_num,  max_sent_len, embedd_size)

        with tf.name_scope('sent_embedd_layers'):
            #注：这里是把bs和max_sent_num合并成一个新的new_bs，也即new_bs=bs*max_sent_num；这样就可以直接输出GRU
            self.word_embedd_resh = tf.reshape(self.word_embedd,     # (bs*max_sent_num, max_sent_len, embedd_size)
                                               [-1, self.max_sent_len, self.embedd_size])
            self.word_bi = self._Bi_GRU_Encoder(self.word_embedd_resh,
                                                name='word_encoder')  # (bs*max_sent_num, max_sent_len, hs*2)
            self.sent_embedd = self._attention(self.word_bi, name='word_attention')  #(bs*max_sent_num, hs*2)

        with tf.name_scope('doc_embedd_layers'):
            # 注：这里把(bs*max_sent_num, hs*2)还原回(bs, max_sent_num, hs*2)
            self.sent_embedd_resh = tf.reshape(self.sent_embedd,
                                               [-1, self.max_sent_num, self.hidden_size * 2]) # (bs, max_sent_num, hs*2)
            self.sent_bi = self._Bi_GRU_Encoder(self.sent_embedd_resh,
                                                name='sent_encoder')  # (bs, max_sent_num, hs*2)
            self.doc_embedd = self._attention(self.sent_bi, name='sent_attention')  # (bs, hs*2)


        with tf.name_scope('output_layers'):
            self.output = layers.fully_connected(inputs=self.doc_embedd, num_outputs=self.num_class,
                                                 activation_fn=None)  #(bs, num_class)
            self.predictions = tf.argmax(self.output, 1, name='predictions')  # (bs, )


        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
            self.optim = tf.train.AdamOptimizer(learning_rate=0.02).minimize(self.loss)


        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')


    def _Bi_GRU_Encoder(self, inputs, name):
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)  #指定每个cell的输出
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=self._length(inputs),
                                                                                 dtype=tf.float32)
            biGRU_output = tf.concat((fw_outputs, bw_outputs), 2)   #(bs, seq_len, hidden*2)
            return biGRU_output

    def _length(self, sequences):   #能不能告诉我，这段代码是干嘛的！！！！！
        # 返回一个序列中每个元素的长度
        used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
        seq_len = tf.reduce_sum(used, reduction_indices=1)
        return tf.cast(seq_len, tf.int32)


    def _attention(self, inputs, name):
        with tf.variable_scope(name):
            # inputs->(bs, seq_len, hs * 2)
            W = tf.Variable(tf.truncated_normal([inputs.shape.as_list()[2], inputs.shape.as_list()[2]],
                                                stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[inputs.shape.as_list()[2]]), name='b')
            inputs_reshape = tf.reshape(inputs, shape=[-1, inputs.shape.as_list()[2]])  #(bs * seq_len, hs * 2)
            u_hidden = tf.nn.tanh(tf.matmul(inputs_reshape, W) + b)  #(bs * seq_len, hs * 2)
            u_hidden = tf.reshape(u_hidden, shape=[-1, inputs.shape.as_list()[1],
                                                   inputs.shape.as_list()[2]],name='u_hidden')  # (bs, seq_len, hs*2)

            # 上面那一大段代码，可以直接用layers.fully_connected()搞定
            # u_hidden = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)

            u_context = tf.Variable(tf.truncated_normal([inputs.shape.as_list()[2]]),
                                                         name='u_context') #(bs,seq_len,hs*2)
            soft_arr = tf.reduce_sum(tf.multiply(u_hidden, u_context), axis=2, keep_dims=True) #(bs, seq_len, 1)
            alpha = tf.nn.softmax(soft_arr, dim=1)  #(bs, seq_len, 1)

            inputs_alpha = tf.multiply(inputs, alpha)  #(bs, seq_len, hs*2)
            atten_output = tf.reduce_sum(inputs_alpha, axis=1)  #(bs, hs*2)
            return atten_output



class HAN_train_test(HAN_model):
    def train(self, data_list, is_save=True):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())  # 随机初始化变量
        tf.summary.FileWriter(self.model_path + self.graph_name, sess.graph)  # 打开记录图

        best_acc = 0
        for i_epoch in range(self.n_epoch):
            print('epech:', i_epoch, '>' * 50)

            X_tra = data_list[0]
            y_tra = data_list[1]
            for bs_X, bs_y, i_bat in self._shuffle_batch_iter(X_tra, y_tra, self.batch_size):
                feed_dict_tra = {self.input_x: bs_X,
                                 self.input_y: bs_y,
                                 self.dropout_keep_prob: 0.2}
                _, loss, tra_acc = sess.run([self.optim, self.loss, self.accuracy], feed_dict=feed_dict_tra)

                if i_bat % self.display == 0:
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
                        for bs_X, bs_y, _ in self._shuffle_batch_iter(X_val, y_val, self.batch_size):
                            feed_dict_val = {self.input_x: bs_X,
                                             self.input_y: bs_y,
                                             self.dropout_keep_prob: 1}
                            bat_val_acc = sess.run(self.accuracy, feed_dict=feed_dict_val)
                            n_corr += (bat_val_acc * len(bs_X))
                        val_acc = n_corr / len(X_val)

                        print('loss:{} >>> train_acc:{} >>> val_acc{}'.format(loss, tra_acc, val_acc))
                        if (val_acc > best_acc) and is_save:
                            best_acc = val_acc
                            self.saver.save(sess, self.model_path + self.model_name)
                            print('best_acc:{} >>> model had save'.format(best_acc))


    def test(self, test_data):
        graph = tf.get_default_graph()  # 通用的，copy过去便可
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

        X_test = test_data[0]
        y_test = test_data[1]

        input_x = graph.get_tensor_by_name("input_x:0")
        input_y = graph.get_tensor_by_name("input_y:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
        accuracy = graph.get_tensor_by_name("accuracy/accuracy:0")
        predictions = graph.get_tensor_by_name("output_layers/predictions:0")

        n_corr = 0
        y_pred = []
        for bs_X, bs_y, _ in self._shuffle_batch_iter(X_test, y_test, self.batch_size, is_shuffle=False):

            feed_dict_test = {input_x: bs_X,
                              input_y: bs_y,
                              dropout_keep_prob: 1.0}

            bat_test_acc = sess.run(accuracy, feed_dict=feed_dict_test)
            bat_y_pred = sess.run(predictions, feed_dict=feed_dict_test)
            n_corr += (bat_test_acc * len(bs_X))
            y_pred += list(bat_y_pred)

        test_acc = n_corr / len(X_test)
        print('test_acc:', test_acc)

        return y_pred

    def _shuffle_batch_iter(self, X, y, bs, is_shuffle=True):
        data_size = len(X)
        # n_batch = int(data_size / bs) if data_size % bs == 0 else int(data_size / bs) + 1
        n_batch = int(data_size / bs)

        if is_shuffle:
            idx_shuffle = np.random.permutation(np.arange(data_size))
            X_shuffle = np.array([X[i] for i in idx_shuffle])
            y_shuffle = np.array([y[i] for i in idx_shuffle])
        else:
            X_shuffle = X
            y_shuffle = y

        for i_bat in range(n_batch):
            sta_idx = i_bat * bs
            end_idx = min((i_bat + 1) * bs, data_size)
            yield X_shuffle[sta_idx: end_idx], y_shuffle[sta_idx: end_idx], i_bat