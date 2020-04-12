

##命名实体识别######################################################################################################
def biLstm_crf_model():   #Keras可以这么方便地实现这个模型
    #先要安装大神写的包：pip install git+https://www.github.com/farizrahman4u/keras-contrib.git

    input_layer = Input(shape=(MAX_SEQ_LEN, ), dtype='int32')
    embedding_layer = Embedding(input_dim=ONEHOT_DIM, output_dim=EMBED_DIM)(input_layer)  
    biLstm_layer = Bidirectional(LSTM(units=LSTM_DIM, return_sequences=True))(embedding_layer)
    Dense_layer = Dense(units=N_TAG)(biLstm_layer)
    crf_layer = CRF(N_TAG, sparse_target=True)(Dense_layer)

    model = Model(input=[input_layer], output=[crf_layer])
    model.compile(loss=crf_layer.loss_function, optimizer='adam', metrics=[crf_layer.accuracy])
    return model




##语义相似度计算###############################################################################################
def textSimilar_model():  #“yin叔”自己定义的模型，是很牛逼了
    MAX_SEQ_LEN = 40
    EMBED_DIM = 300

    #input layer
    input1 = Input(shape=(MAX_SEQ_LEN, ), dtype='int32')
    input2 = Input(shape=(MAX_SEQ_LEN, ), dtype='int32')

    
    #share encoder layer
    text_input = Input(shape=(MAX_SEQ_LEN, ), dtype='int32')
    embedding_layer = Embedding(input_dim=ONEHOT_DIM, output_dim=EMBED_DIM, trainable=True)
    x = embedding_layer(text_input)  #(bs, 40, 300)
    x = TimeDistributed(Dense(units=150, activation='relu'))(x)  #(bs, 40, 150)

    xlstm = CuDNNLSTM(units=150, return_sequences=True)(x)  #(bs, 40, 150)
    xlstm1 = GlobalMaxPooling1D()(xlstm)  #(bs, 1, 150)
    xa = concatenate([xlstm, x])  #(bs, 40, 300)

    xconv1 = Convolution1D(filters=100, kernel_size=1, padding='same', 
                           activation='relu')(xa)
    xconv1 = GlobalMaxPooling1D()(xconv1)  #(bs, 1, 100)

    xconv2 = Convolution1D(filters=100, kernel_size=2, 
                           padding='same', activation='relu')(xa)
    xconv2 = GlobalMaxPooling1D()(xconv2)  #(bs, 1, 100)

    xconv3 = Convolution1D(filters=100, kernel_size=3, 
                           padding='same', activation='relu')(xa)
    xconv3 = GlobalMaxPooling1D()(xconv3)  #(bs, 1, 100)

    xconv4 = Convolution1D(filters=100, kernel_size=4, 
                           dilation_rate=2, padding='same', activation='relu')(xa)
    xconv4 = GlobalMaxPooling1D()(xconv4)  #(bs, 1, 100)

    xconv5 = Convolution1D(filters=100, kernel_size=5, 
                           dilation_rate=2, padding='same', activation='relu')(xa)
    xconv5 = GlobalMaxPooling1D()(xconv5)  #(bs, 1, 100)

    xconv6 = Convolution1D(filters=100, kernel_size=6, 
                           padding='same', activation='relu')(xa)
    xconv6 = GlobalMaxPooling1D()(xconv6)  #(bs, 1, 100)

    xgru = CuDNNGRU(units=300, return_sequences=True)(xa)  #(bs, 40, 300)

    x = concatenate([xconv1, xconv2, xconv3, xconv4, xconv5, xconv6, xlstm1])  #(bs, 1, 750)
    x = Dropout(0.5)(x)
    x = Dense(units=100)(x)
    xout = PReLU()(x)  #(bs, 1, 100)

    text_encoder = Model(inputs=text_input, outputs=[xout, xlstm, xgru])

    
    #interaction layer
    l1, l2, l3 = text_encoder(input1)  #(bs,1,100), (bs,40,150), (bs,40,300)
    r1, r2, r3 = text_encoder(input2)

    diff = subtract([l1, r1])  #(bs, 1, 100)
    mul = multiply([l1, r1])  #(bs, 1, 100)

    cross1 = Dot(axes=[2,2], normalize=True)([l2, r2])  #(bs,40,40)
    cross1 = Reshape((-1, ))(cross1)  #(bs, 1, 1600)
    cross1 = Dropout(0.5)(cross1)
    cross1 = Dense(200)(cross1)
    cross1 = PReLU()(cross1)  #(bs, 1, 200)

    cross2 = Dot(axes=[2,2], normalize=True)([l3, r3])  #(bs,40,40)
    cross2 = Reshape((-1, ))(cross2)  #(bs, 1, 1600)
    cross2 = Dropout(0.5)(cross2)
    cross2 = Dense(200)(cross2)
    cross2 = PReLU()(cross2)  #(bs, 1, 200)

    x = concatenate([l1, r1, diff, mul, cross1, cross2]) #(bs, 1, 800)

    x = BatchNormalization()(x)

    x = Dense(100)(x)
    x = PReLU()(x)
    x = Dropout(0.2)(x)  #(bs, 1, 100)

    x = Dense(50)(x)
    x = PReLU()(x)
    x = Dropout(0.2)(x)  #(bs, 1, 50)

    out = Dense(1, activation='sigmoid')(x)  #(bs, 1, 1)
    
    model = Model(inputs=[input1, input2], outputs=out)
    
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

