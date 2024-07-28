import os
import sys
import re
import nltk
from nltk.corpus import stopwords
from src.exception import CustomException
from src.logger import logging
import tensorflow
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing import sequence


# cleaning data
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)         # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # Remove special characters and digits
    text = text.lower()                       # Convert to lowercase
    return text

# removing stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# word embedding
def one_hot_encoding(review_list):
    voc_size = 5000
    one_hot_representation = [one_hot(words,voc_size) for words in review_list]
    return one_hot_representation

# padding
def padding(one_hot_representation):
    max_len = 500
    padded_one_hot_representation  =  sequence.pad_sequences(one_hot_representation,maxlen=max_len)
    return padded_one_hot_representation