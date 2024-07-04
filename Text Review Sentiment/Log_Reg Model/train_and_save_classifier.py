
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


x_train_df = pd.read_csv('x_train.csv')
y_train_df = pd.read_csv('y_train.csv')

x_train_merge = x_train_df['website_name'] + ' ' + x_train_df['text']
y_train = y_train_df['is_positive_sentiment']

vectorizer = CountVectorizer(lowercase=True, stop_words='english')
vectorizer.fit(x_train_merge)
x_train = vectorizer.transform(x_train_merge)

classifier = LogisticRegression(max_iter = 1200, C=1)
classifier.fit(x_train, y_train)

# import string

# def remove_punctuation(text):
#     return ''.join([char for char in text if char not in string.punctuation])
    

# vectorizer_2 = TfidfVectorizer(ngram_range=(1,1), lowercase=True, stop_words='english', preprocessor=remove_punctuation, min_df=2)
vectorizer_2 = TfidfVectorizer(ngram_range=(1,1), lowercase=True, stop_words='english', min_df=2)
vectorizer_2.fit(x_train_merge)
x_train_2 = vectorizer_2.transform(x_train_merge)

classifier_2 = RandomForestClassifier(max_depth=150)
classifier_2.fit(x_train_2, y_train)

# save classifiers to pkl files
with open('p1_classifier.pkl','wb') as f:
    pickle.dump(classifier,f)

with open('p2_classifier.pkl','wb') as f:
    pickle.dump(classifier_2,f)



