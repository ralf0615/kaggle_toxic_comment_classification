#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:09:23 2018

@author: yuchenli
@content: toxic commment classification from Kaggle
"""

import nltk
nltk.download()
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# Load data
train = pd.read_csv('/Users/yuchenli/Google Drive/Code/Python/'
                    'kaggle_toxic_comment_classification/data/train.csv')

test = pd.read_csv('/Users/yuchenli/Google Drive/Code/Python/'
                    'kaggle_toxic_comment_classification/data/test.csv')

# EDA
train.head()

# TF-IDF (term frequency - inverse document frequency)
tf_idf = TfidfVectorizer(min_df = 1, max_features = None, 
                         strip_accents='unicode', analyzer='word', 
                         token_pattern=r'\w{1,}', ngram_range=(1, 3), 
                         use_idf=1,smooth_idf=1, sublinear_tf=1, 
                         stop_words = 'english')

# Fitting TF-IDF to training set
tf_idf.fit(list(train))
train_tfv =  tf_idf.transform(train)