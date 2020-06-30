import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import scipy as sc

import pickle

import warnings
warnings.filterwarnings("ignore")


mystem = Mystem()
russian_stopwords = stopwords.words("russian")

# Preprocess function
def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and token.strip() not in punctuation
              and not any(map(str.isdigit, token))
              and any(map(str.isalpha, token))
              and len(token) >= 3]

    text = " ".join(tokens)

    return text, tokens


if __name__ == '__main__':
    # Load all models.
    with open('models/counter_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    with open('models/tfidf_vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    with open('models/logreg.pkl', 'rb') as file:
        clf_logreg = pickle.load(file)

    with open('models/naivebayes.pkl', 'rb') as file:
        clf_nb = pickle.load(file)

    # Здесь надо каким-либо удобным способом загрузить нужные документы.
    text_unknown = [['куплю бмв третей серии, до 150 тыс рублей'],
                 ['привет, а вот я хочу найти программиста \
который сможет сделать для меня сайт.'],
                    ['ребенку (7 класс) нужен репетитор по английскому языку'],
                    ['сниму жильё рядом с м. Пушкинская, недорого'],
                    ['есть кто сможет сделать маникюр завтра днем?']]

    # Text preprocessing.
    arr_unknown = []
    
    dict_themes = {0: 'avto', 1: 'dizajn', 2: 'nedvizhimost', 3: 'it', 4: 'photo',
                   5: 'repetitor', 6: 'stil', 7: 'strojka', 8: 'urist', 9: 'yuridicheskie'}
    for i, row in enumerate(text_unknown):
        arr_unknown.append(preprocess_text(row[0])[0])

    # Tokenizing.
    X_1 = tfidf_vectorizer.transform(arr_unknown)
    X_2 = vectorizer.transform(arr_unknown)
    X_unknown = sc.sparse.hstack([X_1, X_2])

    # Predict classes.
    y_predict = np.argmax((1.5 * clf_nb.predict_proba(X_unknown) + clf_logreg.predict_proba(X_unknown)), axis=1)
    y_predict = list(map(lambda x: dict_themes[x], list(y_predict)))
    print(y_predict)

    # Выявление аномалий.
    prob = np.max((1.5 * clf_nb.predict_proba(X_unknown) + clf_logreg.predict_proba(X_unknown)) / 2.5, axis=1)
    anomalies = np.where(prob < 0.4, 1, 0)
    print(np.where(prob < 0.4))
    if np.where(prob < 0.4)[0].shape[0] != 0:
        print(np.array(X_unknown)[np.where(prob < 0.4)[0]])
    else:
        print('There is no anomalies')
