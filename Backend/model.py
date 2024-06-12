from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle




class SGDModel(object):

    def __init__(self):
        self.sgd = SGDClassifier(loss='modified_huber')
        self.vectorizer = TfidfVectorizer()

    def vectorizer_fit(self, X):
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.sgd.fit(X, y)

    def predict_proba(self, X):
        y_proba = self.sgd.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        y_pred = self.sgd.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='Backend/lib/models/TFIDFVectorizer.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))

    def pickle_sgd(self, path='Backend/lib/models/SentimentClassifier.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.sgd, f)
            print("Pickled classifier at {}".format(path))




