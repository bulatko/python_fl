from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

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
