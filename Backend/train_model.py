from model import SGDModel
import pandas as pd
from sklearn.model_selection import train_test_split







import re

def normalize(text):
    text=text.lower()
    text=re.sub(r'[^\w\s]', '', text)
    text=re.sub(r'\d', '', text)
    text=re.sub(r'\s+', ' ', text)
    text=text.strip()
    return text 




def train_model():
    model = SGDModel()
    train_data=pd.read_csv("Backend/lib/data/train.csv")

    train_data_filtered=train_data.dropna()
    X=train_data_filtered.drop(['movieid', 'reviewerName','isFrequentReviewer', 'sentiment'], axis=1)
    y=train_data_filtered['sentiment']
    print(X.shape)

    X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y, test_size=0.1,random_state=10)
    X_train['reviewText']=X_train['reviewText'].apply(lambda x: normalize(x))
    X_test['reviewText']=X_test['reviewText'].apply(lambda x: normalize(x))
    print(X_train.shape)

    model.vectorizer_fit(X_train['reviewText'])
    print('Vectorizer fit complete')

    X_transformed = model.vectorizer_transform(X_train['reviewText'])
    print('Vectorizer transform complete')
    print(X_transformed.shape, y_train.shape)



    model.train(X_transformed, y_train)
    print('Model training complete')

    model.pickle_sgd()
    model.pickle_vectorizer()


if __name__ == "__main__":
    train_model()