import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import nltk
from gensim.models.word2vec import Word2Vec
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, make_scorer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from collections import defaultdict
from sklearn.cluster import KMeans
import pickle


class DataPrepocesser2():
    def __init__(self):
        stop = stopwords.words('english')
        self.tfidf_embedder = TfidfVectorizer(min_df=5, max_df=0.4, stop_words=stop, ngram_range=(1, 2))
        self.red = TruncatedSVD(n_components=100)
        
    def fit(self, X_train): 
        X_red = self.tfidf_embedder.fit_transform(X_train)
        self.red.fit(X_red)
        
    def transform(self, X):
        X2 = self.red.transform(self.tfidf_embedder.transform(X))    
        return X2
    
    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)

def text_prepocessing(text):
    stemmer = EnglishStemmer()
    text = re.sub('[\W]', ' ', text)
    text = re.sub('\ +', ' ', text)
    text = text.lower()
    text = [stemmer.stem(word) + ' ' for word in text.split(' ') if len(word) > 1]
    s = ''
    for t in text:
        s += t
    return s[:-1]

def predict_score(text):
    text = np.array([text_prepocessing(text)])

    prep = pickle.load(open('preproc.pkl', 'rb'))
    clf = pickle.load(open('clf.pkl', 'rb'))
    clusterizer = pickle.load(open('clusterizer.pkl', 'rb'))

    X_test_v = prep.transform(text)
    print(X_test_v.shape)

    unsup_ans = clusterizer.predict(X_test_v)

    unsup_ans = pd.get_dummies(unsup_ans).values
    
    ans = np.zeros(8)
    ans[unsup_ans] = 1
    X_test_v = np.concatenate((X_test_v, ans[np.newaxis]), axis=1)
    print(X_test_v.shape, unsup_ans.shape)
    pred = clf.predict(X_test_v)
    
    return pred[0]


while 1:
    print("Do u want to predict comment score? (y/n)")
    p = input()
    if p == 'y':
        text = input()
        pred = predict_score(text)
        print("score is: {}".format(pred))
    elif p == 'n':
        print("program ended")
        break
