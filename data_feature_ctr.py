
import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm
from scipy.spatial.distance import cosine,cityblock,jaccard

train_data=pd.read_csv("data/df_train_seg.csv")
test_data=pd.read_csv("data/df_test_seg.csv")

def sent2vec(s):
    words=str(s).split()
    M=[]
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M=np.array(M)
    v=M.sum(axis=0)
    return v/np.sqrt((v**2).sum())

model=gensim.models.KeyedVectors.load_word2vec_format("model/model_word_vec_200.txt",binary=False)

def build_feature(data):
    X = pd.DataFrame()
    question1_vector = np.zeros((data.shape[0], 200))
    for i, q in tqdm(enumerate(data["seg_prefix"].values)):
        question1_vector[i, :] = sent2vec(q)

    question2_vector = np.zeros((data.shape[0], 200))
    for i, q in tqdm(enumerate(data["seg_title"].values)):
        question2_vector[i, :] = sent2vec(q)


    X["cosine_disttance"] = [cosine(x, y) for (x, y) in
                             zip(np.nan_to_num(question1_vector), np.nan_to_num(question2_vector))]
    prediction_cols = ['seg0', 'seg1', 'seg2', 'seg3', 'seg4','seg5', 'seg6', 'seg7', 'seg8', 'seg9']
    for j,col in enumerate(prediction_cols):
        question1_vector = np.zeros((data.shape[0], 200))
        for i, q in tqdm(enumerate(data[col].values)):
            question1_vector[i, :] = sent2vec(q)

        X[col + "_cosine"] = [cosine(x, y) for (x, y) in
                              zip(np.nan_to_num(question1_vector), np.nan_to_num(question2_vector))]
        X["score_cos_mul_"+str(j)]=X[col + "_cosine"]*data["score"+str(j)]
    return X

df_train=build_feature(train_data)
df_test=build_feature(test_data)

df_train.to_csv("feature/feature_cosine_score_mul_train.csv",index=False)
df_test.to_csv("feature/feature_cosine_score_mul_test.csv",index=False)
