import pandas as pd


# train_data = pd.read_table('data/oppo_round1_train_20180929.txt',
#         names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)
train_data = pd.read_csv('data/train_data_split.csv')
val_data = pd.read_table('data/oppo_round1_vali_20180929.txt',
        names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8').astype(str)
test_data = pd.read_table('data/oppo_round1_test_A_20180929.txt',
        names = ['prefix','query_prediction','title','tag'],header = None, encoding='utf-8').astype(str)
#train_data = train_data[train_data['label'] != '音乐' ].reset_index()
test_data['label'] = -1
#train_data.drop(['index'],axis=1,inplace=True)

print("Starting LightGBM. Train shape: {},val shape: {}, test shape: {}".format(train_data.shape, val_data.shape,test_data.shape))

train_data = pd.concat([train_data,val_data])
train_data['label'] = train_data['label'].apply(lambda x: int(x))
test_data['label'] = test_data['label'].apply(lambda x: int(x))

#title in query_pre
data=pd.concat([train_data,test_data])
data['title_in_query_ca']=data.apply(lambda x:1 if x.title in x.query_prediction else 0,axis=1)
import re 
import json


data['title_in_query_ca']=data.apply(lambda x:1 if x.title in x.query_prediction else 0,axis=1)
data['prefix_in_title_ca']=data.apply(lambda x:1 if x.prefix in x.title else 0,axis=1)

def  get_title_value(title,query):
#     print(title,query)
    query=str(query)
    js=json.loads(query)
    try:
        value=js[title]
    except:
        value=0
    
    return float(value)

def  title_is_max_value(title,query):
#     print(title,query)
    query=str(query)
    js=json.loads(query)
    try:
        value=float(js[title])
    except:
        value=0
        return 0 
    values=js.values()
    values=[float(i) for i in  values]
    max_value=max(values)
    if  abs(max_value-value)<0.0001:
        return 1
    
    return 0

def  get_query_num(title,query):
#     print(title,query)
    query=str(query)
    js=json.loads(query)
    if type(js)!=dict:
        return 0
    
    return len(js.keys())

def  get_max_value(title,query):
#     print(title,query)
    query=str(query)
    js=json.loads(query)
    if type(js)!=dict:
        return 0
    values=js.values()
    values=[float(i) for i in  values]
    if len(values)==0:
        return 0
    
    max_value=max(values)
    
    return max_value

def  get_second_value(title,query):
#     print(title,query)
    query=str(query)
    js=json.loads(query)
    if type(js)!=dict:
        return 0

    values=js.values()
    values=[float(i) for i in  values]
    if len(values)<2:
        return 0
    
    sort_values=sorted(values,reverse=True)
    
    return  sort_values[1]


def  get_min_value(title,query):
#     print(title,query)
    query=str(query)
    js=json.loads(query)
    if type(js)!=dict:
        return 0

    values=js.values()
    values=[float(i) for i in  values]
    if len(values)==0:
        return 0
    
    return  min(values)

def  count_title_in_query(title,query):
#     print(title,query)
    query=str(query)
    js=json.loads(query)
    if type(js)!=dict:
        return 0

    keys=js.keys()
    count=0
    for i in keys:
        if title in i:
            count+=1
    
    return  count

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
    
def include_ch(prefix):
    s=re.search("[\u4e00-\u9fa5]+",prefix)
    
    if s!=None:
        return 1
    else:
        return 0
    
data['title_in_query_value_ismax_ca']=data.apply(lambda x:title_is_max_value(x.title,x.query_prediction),axis=1)
data['query_num']=data.apply(lambda x:get_query_num(x.title,x.query_prediction),axis=1)
data['title_in_query_value_num']=data.apply(lambda x:get_title_value(x.title,x.query_prediction),axis=1)
data['query_max_value_num']=data.apply(lambda x:get_max_value(x.title,x.query_prediction),axis=1)
data['query_second_value_num']=data.apply(lambda x:get_second_value(x.title,x.query_prediction),axis=1)
data['query_min_value_num']=data.apply(lambda x:get_min_value(x.title,x.query_prediction),axis=1)
data['maxV_diff_secondV_num']=data['query_max_value_num']-data['query_second_value_num']
data['maxV_diff_minV_num']=data['query_max_value_num']-data['query_min_value_num']
data['title_in_query_count_num']=data.apply(lambda x:count_title_in_query(x.title,x.query_prediction),axis=1)
data['prefix_inc_num_ca']=data.apply(lambda x:include_num(x.prefix),axis=1)
data['prefix_inc_zm_ca']=data.apply(lambda x:include_zm(x.prefix),axis=1)
data['prefix_inc_ch_ca']=data.apply(lambda x:include_ch(x.prefix),axis=1)

data['prefix_equal_title_ca']=data.apply(lambda x: 1 if x.prefix==x.title else 0,axis=1)
data['prefix_dev_title_num']=data.apply(lambda x: len(x.prefix)/len(x.title),axis=1)

data['title_len_num']=data['title'].map(lambda x:len(x))
data['prefix_len_num']=data['prefix'].map(lambda x:len(x))
data['prefix_diff_title_len_num']=data['title_len_num']-data['prefix_len_num']

from tqdm import tqdm_notebook
origin_cate_list=['prefix','query_prediction', 'title', 'tag',]

for i in tqdm_notebook(origin_cate_list):

    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    #热度
    gp=data.groupby(i).agg('size').reset_index().rename(columns={0:'%s_hot_num'%i})
    data=pd.merge(data,gp,how='left',on=i)
    
    
#count 特征
train_data=data[data["label"]!=-1]
test_data=data[data["label"]==-1]
items = ['tag','prefix','query_prediction','title']

for item in items:
    temp = train_data.groupby(item).agg('size').reset_index().rename(columns={0:'%s_count_num'%item})
    train_data = pd.merge(train_data, temp, on=item, how='left')
    test_data = pd.merge(test_data, temp, on=item, how='left')
    
for i in range(len(items)):
    for j in range(i+1, len(items)):
        item_g = [items[i], items[j]]
        new_fea_name='_'.join(item_g)
        temp = train_data.groupby(item_g).agg('size').reset_index().rename(columns={0:'%s_count_num'%new_fea_name})
        train_data = pd.merge(train_data, temp, on=item_g, how='left')
        test_data = pd.merge(test_data, temp, on=item_g, how='left')

num_feature=[i for i in train_data.columns if "_ctr" in i or "_num" in i]
feature=num_feature
print(feature,(len(feature)))
print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_data[feature].shape, test_data[feature].shape))
train_data[feature].to_csv("feature/new_feature_ration_train.csv",index=False)
test_data[feature].to_csv("feature/new_feature_ration_test.csv",index=False)
