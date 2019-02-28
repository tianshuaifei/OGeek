# -*- coding: utf-8 -*-


import lightgbm as lgb
import pandas as pd
import pickle
import gc



import numpy as np

train_data=pd.read_csv("data/df_train.csv")
val_data=pd.read_csv("data/df_val.csv")
test_data=pd.read_csv("data/df_test.csv")
print("load data. train_data shape: {}, val_data shape: {},test_data shape: {}".format(train_data.shape,val_data.shape, test_data.shape))


print(train_data.columns)

num_feature_name=['prediction_num', 'score0', 'score1']

##############################################
data=pd.concat([train_data,val_data,test_data])
def add_data(df):
    df["pall"]=df["seg0"]+" "+df["seg1"]+" "+df["seg2"]+" "+df["seg3"]+" "+df["seg4"]+" "+ df["seg5"]+\
                  " "+df["seg6"]+" "+df["seg7"]+" "+df["seg8"]+" "+df["seg9"]
    df["all_all"] = df["prefix"] + df["title"] + df["tag"]
    df["title_tag"] = df["title"] + df["tag"]
    return df

data=add_data(data)
#################################################
import re
def include_num(prefix):
    s=re.search("\d",prefix)
    if s!=None:
        return 1
    else:
        return 0

def include_zm(prefix):
    s=re.search("[a-zA-Z]",prefix)
    if s!=None:
        return 1
    else:
        return 0

def build_feature(data):
    data['len_q1'] = data.prefix.apply(lambda x: len(str(x)))
    data['len_q2'] = data.title.apply(lambda x: len(str(x)))
    data['diff_len'] = data.len_q2 - data.len_q1
    data['seg_prefix_len'] = data["seg_prefix"].apply(lambda x: len(str(x).split()))
    data['seg_title_len'] = data["seg_title"].apply(lambda x: len(str(x).split()))
    data['diff_seg_len'] = data["seg_title_len"] - data['seg_prefix_len']

    data["prefix_is_title"]=data.apply(lambda x:1 if x.title.lower()==x.prefix.lower() else 0,axis=1)
    data["prefix_isbegin_title"] = data.apply(lambda x: 1 if x.title.lower().startswith(x.prefix.lower()) and x.prefix_is_title==0 else 0, axis=1)
    data["prefix_in_title"] = data.apply(
        lambda x: 1 if x.prefix.lower() in x.title.lower()  and x.prefix_is_title == 0 and x.prefix_isbegin_title==0 else 0, axis=1)
    data["title_is_predictions"] = data.apply(lambda x: 1 if x.title.lower() == x.text0.lower() else 0, axis=1)
    data["it_is_true"] = data.apply(lambda x: 1 if x.prefix_is_title == 1 and x.title_is_predictions == 1 else 0, axis=1)
    data["it_is_true2"] = data.apply(lambda x: 1 if x.prefix_isbegin_title == 1 and x.title_is_predictions == 1 else 0,
                                    axis=1)
    data["prefix_is_predictions"] = data.apply(lambda x: 1 if x.prefix.lower() == x.text0.lower() else 0, axis=1)
    data['prefix_inc_num_ca'] = data.apply(lambda x: include_num(x.prefix), axis=1)
    data['prefix_inc_zm_ca'] = data.apply(lambda x: include_zm(x.prefix), axis=1)
    #data['prefix_inc_zm_bd'] = data.apply(lambda x: include_bd(x.prefix), axis=1)
    return data
num_feature_name.append("len_q1")
num_feature_name.append("len_q2")
num_feature_name.append("diff_len")
num_feature_name.append("seg_prefix_len")
num_feature_name.append("seg_title_len")
num_feature_name.append("diff_seg_len")
num_feature_name.append("prefix_is_title")
num_feature_name.append("prefix_isbegin_title")
num_feature_name.append("prefix_in_title")
num_feature_name.append("title_is_predictions")
num_feature_name.append("it_is_true")
num_feature_name.append("it_is_true2")
num_feature_name.append("prefix_is_predictions")
num_feature_name.append("prefix_inc_num_ca")
num_feature_name.append("prefix_inc_zm_ca")

data=build_feature(data)

temp=data.groupby(["title","tag"]).size().reset_index().rename(columns={0:"title_tag_count"})
data=data.merge(temp,'left',on=["title","tag"])

temp=data.groupby(["title"])["tag"].count().reset_index().rename(columns={"tag":"title_count_tag"})
data=pd.merge(data,temp,how='left',on=["title"])

temp=data.groupby(["title"])["tag"].nunique().reset_index().rename(columns={"tag":"title_nunique_tag"})
data=pd.merge(data,temp,how='left',on=["title"])
num_feature_name.append("title_tag_count")
num_feature_name.append("title_count_tag")
num_feature_name.append("title_nunique_tag")



data['title_len'] = data['prefix'].apply(len)
data['title_len'] = data['title_len'].map(lambda x: 30 if x>30 else x)

data['prefix_len'] = data['prefix'].apply(len)
data['prefix_len'] = data['prefix_len'].map(lambda x: 8 if x>8 else x)

from sklearn import preprocessing
label_columns=['prefix','title', 'tag']
for col in label_columns:
    le = preprocessing.LabelEncoder()
    le.fit(data[col].values)
    data[col]=le.transform(data[col].values)
num_feature_name=num_feature_name+label_columns

train_data_df = data[data['label']!=-1]
test_data_df = data[data['label']==-1]
val_df=train_data_df.iloc[1999998:,:]
train_df=train_data_df.iloc[:1999998,:]
test_df=test_data_df
##################################################################################################

items = ['tag','title_len','prefix_len']
for item in items:
    temp = train_df.groupby(item, as_index = False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
    temp[item+'_ctr'] = (temp[item+'_click'])/(temp[item+'_count'])
    temp[item + '_ctr'] = temp.apply(lambda x:np.nan if x[item+'_count']<2 else x[item + '_ctr'],axis=1)
    temp[item+'_click']=np.log1p(temp[item+'_click'])
    temp[item +'_count'] = np.log1p(temp[item + '_count'])
    train_df = pd.merge(train_df, temp, on=item, how='left')
    val_df = pd.merge(val_df, temp, on=item, how='left')
    test_df = pd.merge(test_df, temp, on=item, how='left')

num_feature_name.append("prefix_len_ctr")
num_feature_name.append("title_len_ctr")
num_feature_name.append("tag_ctr")

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.2, reg_lambda=0.9, max_depth=-1,
                                   n_estimators=50000, objective='binary', subsample=0.90, colsample_bytree=0.8,
                                   subsample_freq=1, learning_rate=0.05, random_state=2018, n_jobs=16,
                                   min_child_weight=2, min_child_samples=5)

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
feats = num_feature_name
print(len(feats),feats)

for k, (train_in, test_in) in enumerate(skf.split(train_df[feats], train_df['label'])):
    print('train _K_ flod', k)
    X_train, y_train = train_df[feats].iloc[train_in], train_df['label'].iloc[train_in]
    X_test, y_test = train_df[feats].iloc[test_in], train_df['label'].iloc[test_in]


    lgb_model.fit(X_train, y_train, verbose=50, eval_set=[(X_train, y_train), (X_test, y_test)],
                  early_stopping_rounds=50)

    print(f1_score(y_test,np.where(lgb_model.predict_proba(X_test, num_iteration=lgb_model.best_iteration_)[:, 1] > 0.5, 1,0)))
    print(f1_score(val_df["label"],np.where(lgb_model.predict_proba(val_df[feats], num_iteration=lgb_model.best_iteration_)[:, 1] > 0.5, 1,0)))
    print(f1_score(val_df["label"],
                   np.where(lgb_model.predict_proba(val_df[feats], num_iteration=lgb_model.best_iteration_)[:, 1] > 0.4,
                            1, 0)))
    print(f1_score(val_df["label"],
                   np.where(lgb_model.predict_proba(val_df[feats], num_iteration=lgb_model.best_iteration_)[:, 1] > 0.35,
                            1, 0)))
    print(f1_score(val_df["label"], np.where(
        lgb_model.predict_proba(val_df[feats], num_iteration=lgb_model.best_iteration_)[:, 1] > 0.3, 1, 0)))


