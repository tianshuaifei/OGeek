

import pandas as pd
import numpy as np
from keras import Input,layers,Model,optimizers
from keras.layers import Embedding,Bidirectional,CuDNNGRU,Dropout,Dense,BatchNormalization,concatenate


def build_model():
    Max_num = 220000
    Q_length = 9
    A_length = 35
    P_length = 10

    # emb_q = Embedding(5000, 100,input_length=20, trainable=True)
    # emb_a = Embedding(5000, 100,input_length=50, trainable=True)
    emb_q = Embedding(Max_num + 1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],
                      input_length=Q_length, trainable=False)
    emb_a = Embedding(Max_num + 1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],
                      input_length=A_length, trainable=False)
    emb_p = Embedding(Max_num + 1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],
                      input_length=P_length, trainable=False)

    seq1 = Input(shape=(Q_length,))
    seq2 = Input(shape=(A_length,))
    emb1 = emb_q(seq1)
    emb2 = emb_a(seq2)
    lstm_layer1 = Bidirectional(CuDNNGRU(64, return_sequences=True))
    lstm_layer2 = Bidirectional(CuDNNGRU(64))
    que = lstm_layer1(emb1)
    ans = lstm_layer1(emb2)
    que = lstm_layer2(que)
    ans = lstm_layer2(ans)
    pmul = layers.multiply([que, ans])
    psub = layers.subtract([que, ans])

    add_feature = Input(shape=(22,))
    add_x = Dense(16, activation='elu')(add_feature)
    add_x = BatchNormalization()(add_x)

    score_feature = Input(shape=(10,))
    p_a = []
    p_input = []
    for i in range(10):
        seq_p = Input(shape=(P_length,))
        p_input.append(seq_p)
        if i <3:
            emb_ = emb_p(seq_p)
            emb_ = lstm_layer1(emb_)
            pre = lstm_layer2(emb_)
            mul = layers.multiply([pre, ans])
            sub = layers.subtract([pre, ans])
            p_vec = concatenate([mul, sub])
            p_a.append(p_vec)

    p_v = concatenate(p_a)
    p_v = Dense(512, activation='elu')(p_v)
    p_v = Dropout(0.2)(p_v)
    # p_v = Dense(128, activation='elu')(p_v)
    # p_v = Dropout(0.2)(p_v)

    merge_vec = concatenate([pmul, psub, add_x,score_feature,p_v])

    x = BatchNormalization()(merge_vec)
    x = Dense(300, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(300, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(300, activation='elu')(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)
    model_ = Model(inputs=[seq1, seq2, add_feature,score_feature]+p_input, outputs=pred)
    opt = optimizers.Adam(lr=0.001)
    model_.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    #model_.summary()
    return model_




from keras.callbacks import EarlyStopping, ModelCheckpoint


from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
class myMetrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        # print("--------------------------")
        # print(len(self.validation_data))
        # print(self.validation_data[-1])
        # print(self.validation_data[-2])
        # print(self.validation_data[-3])
        # print(self.validation_data[-4])
        # print("--------------------------")
        val_predict = (np.asarray(self.model.predict(self.validation_data[0:-3],batch_size=1024))).round()
        val_targ = self.validation_data[-3]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        print()
        print("val_f1    ", _val_f1)
        print("val_recall  ", _val_recall)
        print("val_precision  ", _val_precision)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return




f_cb=myMetrics()

early_stopping = EarlyStopping(monitor="val_loss", patience=5)
best_model_path = "model/best_t_model_rcnn_weight.h5"
model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)


from sklearn.model_selection import train_test_split


#X_train, X_test,A_train,A_test,F_train,F_test, y_train, y_test = train_test_split(p_train_q, t_train_a,add_feature,label,test_size=0.2, random_state=42)
p_train_q=pd.read_pickle('model/world_p_train_q.pkl')
t_train_a=pd.read_pickle('model/world_t_train_a.pkl')
label=pd.read_pickle('model/label.pkl')
word_embedding_matrix=pd.read_pickle('model/gensim_word_200.pkl')
add_feature=pd.read_pickle('model/add_feature.pkl')
train_list=pd.read_pickle('model/world_p_train.pkl')
train_score=pd.read_pickle('model/train_score.pkl')

print(p_train_q[:5])

data_input=[p_train_q, t_train_a,add_feature,train_score]+train_list
print(label)
train_input=[]
val_input=[]
for data in data_input:
    X_train,X_teat,label_train,label_val=train_test_split(data,label,test_size=0.2,random_state=2018,shuffle=True)
    train_input.append(X_train)
    val_input.append(X_teat)

print(len(data_input))
m_model = build_model()
# m_model.fit([X_train, A_train,F_train], y_train, epochs=100,
#                    batch_size=1024,
#                    validation_split=0.2,
#                    #validation_data=([X_test,A_test,F_test],y_test),
#                    callbacks=[model_checkpoint,early_stopping,f_cb],
#                   shuffle=True)
m_model.fit(train_input, label_train, epochs=100,
                   batch_size=1024,
                   #validation_split=0.2,
                   validation_data=(val_input,label_val),
                   callbacks=[model_checkpoint,early_stopping,f_cb],
                  shuffle=True)
m_model.load_weights(best_model_path)


tq=pd.read_pickle('model/world_p_test_q.pkl')
ta=pd.read_pickle('model/world_t_test_a.pkl')
af=pd.read_pickle('model/add_feature_y.pkl')
test_list=pd.read_pickle('model/world_p_test.pkl')
test_score=pd.read_pickle('model/test_score.pkl')
resu=m_model.predict([tq,ta,af,test_score]+test_list,batch_size=1024)

sub=pd.DataFrame(resu)
sub.to_csv("sub.csv",index=False)
