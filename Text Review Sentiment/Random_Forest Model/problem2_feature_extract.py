import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Given a list of two strings with the first being the website name and second
# being the review, merges them into one string to be used as a feature
# import string

# def remove_punctuation(text):
#     return ''.join([char for char in text if char not in string.punctuation])

def feature_extractor_2(x_text):

    x_train_df = pd.read_csv('x_train.csv')
    x_train_merge = x_train_df['website_name'] + ' ' + x_train_df['text']

    # vectorizer = TfidfVectorizer(ngram_range=(1,1), lowercase=True, stop_words='english', preprocessor=remove_punctuation, min_df=2)
    vectorizer = TfidfVectorizer(ngram_range=(1,1), lowercase=True, stop_words='english', min_df=2)
    vectorizer.fit(x_train_merge)

    x_text_df = pd.DataFrame(x_text, columns=['website_name', 'text'])

    x_test = x_text_df['website_name'] + ' ' + x_text_df['text']
    x = vectorizer.transform(x_test)

    return x






