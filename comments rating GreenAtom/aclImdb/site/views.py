from django.http import HttpResponse
from django.views import View

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
nltk.download('stopwords')
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, make_scorer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from collections import defaultdict
from sklearn.cluster import KMeans
import pickle
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
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, make_scorer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from collections import defaultdict
from sklearn.cluster import KMeans
import pickle
from catboost import CatBoostClassifier
from .prep import DataPrepocesser2

started = 0
path = 'project_name/project_name/'

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)







class IndexView(View):
    def get(self, request, *args, **kwargs):
        return HttpResponse('')


class Scorer(View):
    def __start(self):
        global started, clf, clusterizer, prep
        prep = CustomUnpickler(open(path + 'preproc.pkl', 'rb')).load()
        clf = CustomUnpickler(open(path + 'clf.pkl', 'rb')).load()
        clusterizer = CustomUnpickler(open(path + 'clusterizer.pkl', 'rb')).load()
        started = 1

    def __text_prepocessing(self, text):
        stemmer = EnglishStemmer()
        text = re.sub('[\W]', ' ', text)
        text = re.sub('\ +', ' ', text)
        text = text.lower()
        text = [stemmer.stem(word) + ' ' for word in text.split(' ') if len(word) > 1]
        s = ''
        for t in text:
            s += t
        return s[:-1]

    def __predict_score(self, text):
        text = np.array([self.__text_prepocessing(text)])
        global clf, prep, clusterizer
        X_test_v = prep.transform(text)
        unsup_ans = clusterizer.predict(X_test_v)
        ans = np.zeros(8)
        ans[unsup_ans] = 1
        X_test_v = np.concatenate((X_test_v, ans[np.newaxis]), axis=1)
        pred = clf.predict(X_test_v)
        return pred[0]

    def get(self, request, *args, **kwargs):
        global started


        comment = request.GET.get('comment')
        if comment:
            if not started:
                self.__start()
            score = self.__predict_score(comment)[0]
            text = "Your comment: <p>{}</p><br>Your score: {}<br>Semantic of comment: {}".format(
            comment,
            score,
            "Positive" if score > 5 else "Negative"
            )
            return HttpResponse(text)
