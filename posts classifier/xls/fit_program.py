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
    print(text)
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
    # Load datasets.
    df_avto = pd.read_excel('xls/avto.xls')
    df_dizajn = pd.read_excel('xls/dizajn.xls')
    df_nedvizhimost = pd.read_excel('xls/nedvizhimost.xls')
    df_photo = pd.read_excel('xls/photo.xls')

    df_repetitor = pd.read_excel('xls/repetitor.xls')
    df_strojka = pd.read_excel('xls/strojka.xls')
    df_urist = pd.read_excel('xls/urist.xls')
    df_yuridicheskie = pd.read_excel('xls/yuridicheskie.xls')

    # Проблемные ребята, справился с ними путем того, что сохранил их как .xlsx вместо .xls
    # df_it = pd.read_excel('xls/it.xls')
    # df_stil = pd.read_excel('xls/stil.xls')
    df_it = pd.read_excel('xls/it_v0.xlsx')
    df_stil = pd.read_excel('xls/stil_v0.xlsx')

    nltk.download('punkt')
    nltk.download("stopwords")

    # Make preprocessing of text.
    arr_avto = []
    arr_dizajn = []
    arr_nedvizhimost = []
    arr_it = []
    arr_photo = []

    arr_repetitor = []
    arr_stil = []
    arr_strojka = []
    arr_urist = []
    arr_yuridicheskie = []

    for index, row in df_avto.iterrows():
        arr_avto.append(preprocess_text(row[0])[0])

    for index, row in df_dizajn.iterrows():
        arr_dizajn.append(preprocess_text(row[0])[0])

    for index, row in df_nedvizhimost.iterrows():
        arr_nedvizhimost.append(preprocess_text(row[0])[0])

    for index, row in df_it.iterrows():
        arr_it.append(preprocess_text(row[0])[0])

    for index, row in df_photo.iterrows():
        arr_photo.append(preprocess_text(row[0])[0])

    for index, row in df_repetitor.iterrows():
        arr_repetitor.append(preprocess_text(row[0])[0])

    for index, row in df_stil.iterrows():
        arr_stil.append(preprocess_text(row[0])[0])

    for index, row in df_strojka.iterrows():
        arr_strojka.append(preprocess_text(row[0])[0])

    for index, row in df_urist.iterrows():
        arr_urist.append(preprocess_text(row[0])[0])

    for index, row in df_yuridicheskie.iterrows():
        arr_yuridicheskie.append(preprocess_text(row[0])[0])

    # Make array of all datasets to comfortable usage.
    arr_of_themes = []

    arr_of_themes.append(arr_avto)
    arr_of_themes.append(arr_dizajn)
    arr_of_themes.append(arr_nedvizhimost)
    arr_of_themes.append(arr_it)
    arr_of_themes.append(arr_photo)

    arr_of_themes.append(arr_repetitor)
    arr_of_themes.append(arr_stil)
    arr_of_themes.append(arr_strojka)
    arr_of_themes.append(arr_urist)
    arr_of_themes.append(arr_yuridicheskie)

    dict_themes = {0: 'avto', 1: 'dizajn', 2: 'nedvizhimost', 3: 'it', 4: 'photo',
                   5: 'repetitor', 6: 'stil', 7: 'strojka', 8: 'urist', 9: 'yuridicheskie'}

    arr_full = []

    for i in range(10):
        arr_full.extend(arr_of_themes[i])

    targets = []

    for i in range(10):
        local_target = [i] * len(arr_of_themes[i])
        targets.extend(local_target)

    # Make vectorization using counter and tfidf.
    vectorizer = CountVectorizer(min_df=50)
    vectorizer.fit(arr_full)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_vectorizer.fit([' '.join(arr_of_themes[i]) for i in range(10)])

    X_1 = tfidf_vectorizer.transform(arr_full)
    X_2 = vectorizer.transform(arr_full)
    X = sc.sparse.hstack([X_1, X_2])

    # Fit needed models.
    clf_logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clf_logreg.fit(X, targets)

    clf_nb = MultinomialNB(alpha=0.5)
    clf_nb.fit(X, targets)

    # Save all to files.
    with open('models/counter_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)

    with open('models/tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)

    with open('models/logreg.pkl', 'wb') as file:
        pickle.dump(clf_logreg, file)

    with open('models/naivebayes.pkl', 'wb') as file:
        pickle.dump(clf_nb, file)



