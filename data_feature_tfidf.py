

import numpy as np
import pandas as pd
from collections import Counter
import functools

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

def load_char_weight(data):
    train_qs = pd.Series(data['chars_x'].tolist() + data['chars_y'].tolist())
    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    return weights
def load_word_weight(data):
    train_qs = pd.Series(data['words_x'].tolist() + data['words_y'].tolist())
    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    return weights

def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['words_x']:
        q1words[word] = 1
    for word in row['words_y']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_char_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['chars_x']:
        q1words[word] = 1
    for word in row['chars_y']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def build_features(data,weights_word,weights_char):
    X = pd.DataFrame()
    # X['word_match'] = data.apply(word_match_share, axis=1, raw=True)   # 1
    # X['char_match'] = data.apply(char_match_share, axis=1, raw=True)   # 2
    # X['jaccard_word'] = data.apply(jaccard_word, axis=1, raw=True)     # 3
    # X['jaccard_char'] = data.apply(jaccard_char, axis=1, raw=True)     # 4
    # X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True)               # 5
    # X['char_diff'] = data.apply(char_diff, axis=1, raw=True)           # 6
    # X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True)             # 7
    # X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True)        # 8
    # X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)   # 9
    # X['char_diff_unique'] = data.apply(char_diff_unique, axis=1, raw=True) # 10
    # X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)   # 11
    # X['char_ratio_unique'] = data.apply(char_ratio_unique, axis=1, raw=True)  # 12

    f = functools.partial(tfidf_word_match_share, weights=weights_word)
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True)
    d = functools.partial(tfidf_char_match_share, weights=weights_char)
    X['tfidf_charm'] = data.apply(d, axis=1, raw=True)

    return X

train_data=pd.read_csv("data/df_train_seg.csv")
test_data=pd.read_csv("data/df_test_seg.csv")

train_data['words_x'] = train_data['seg_prefix'].map(lambda x: str(x).split())
train_data['words_y'] = train_data['seg_title'].map(lambda x: str(x).split())
train_data['chars_x'] = train_data['prefix'].map(lambda x: list(x))
train_data['chars_y'] = train_data['title'].map(lambda x: list(x))

test_data['words_x'] = test_data['seg_prefix'].map(lambda x: str(x).split())
test_data['words_y'] = test_data['seg_title'].map(lambda x: str(x).split())
test_data['chars_x'] = test_data['prefix'].map(lambda x: list(x))
test_data['chars_y'] = test_data['title'].map(lambda x: list(x))

datas=pd.concat([train_data,test_data])
weights_word=load_word_weight(datas)
weights_char=load_char_weight(datas)

df_train=build_features(train_data,weights_word,weights_char)
df_test=build_features(test_data,weights_word,weights_char)

df_train.to_csv("feature/feature_tfidf_train.csv",index=False)
df_test.to_csv("feature/feature_tfidf_test.csv",index=False)
