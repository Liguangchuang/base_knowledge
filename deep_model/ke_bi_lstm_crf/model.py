from keras.layers import Embedding, LSTM, Dropout, Dense, Input
from keras.models import Model
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras_contrib.layers import CRF


### 应该说，keras和tf最主要的差别就是这里了吧！
# keras的模型搭建就几行代码就搞定了，tf要很多很多代码
# keras实现训练，有集成的fit和predict函数，直接调用就可以了，tf我喜欢用一个类来写fit和predict程序，所有东西都要自己写
# 它只给了placeholder和sess.run()，其他的fit和predict要自己构建！

# 应该说，keras是更方便的，tf虽然更底层，但其实实现fit和predict就可以满足要求了，写太多无用的代码也浪费时间！



def biLstm_crf_model(args):   #Keras可以这么方便地实现这个模型
    input = Input(shape=(args.maxLen, ), dtype='int32')
    embedding_layer = Embedding(input_dim=len(args.embedd_matrix), output_dim=len(args.embedd_matrix[0]),
                                weights=[args.embedd_matrix], trainable=False)(input)
    biLstm_out = Bidirectional(LSTM(units=args.hidden_size, return_sequences=True))(embedding_layer)
    Dense_out = TimeDistributed(Dense(units=args.num_tags))(biLstm_out)
    crf = CRF(units=args.num_tags, sparse_target=True)  #crf才有loss_function，而crf_out没有要注意
    crf_out = crf(Dense_out)

    model = Model(inputs=[input], outputs=[crf_out])
    model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    model.summary()
    return model


