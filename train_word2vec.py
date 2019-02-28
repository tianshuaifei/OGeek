import pandas as pd
import numpy as np
import jieba
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    return " ".join(sentence_seged)

train_data=pd.read_csv("data/df_train.csv")
test_data=pd.read_csv("data/df_test.csv")

list_seg=["prefix","title","text0","text1","text2","text3","text4","text5","text6","text7","text8","text9"]
data=[]
for seg in list_seg:
    train_data[seg+"_list"] = train_data[seg].apply(lambda x: seg_sentence(x))
    test_data[seg + "_list"] = test_data[seg].apply(lambda x: seg_sentence(x))

    train_data[seg + "_list"] = train_data[seg + "_list"].map(lambda x: str(x).split())
    test_data[seg + "_list"] = test_data[seg + "_list"].map(lambda x: str(x).split())

    data.extend(train_data[seg+"_list"].values.tolist())
    data.extend(test_data[seg + "_list"].values.tolist())


from gensim.models.word2vec import Word2Vec
model=Word2Vec(data,size=200,sg=1,window=5,min_count=1,workers=8,iter=10)
model.wv.save_word2vec_format('model/model_word_vec_200.txt', binary=False)
