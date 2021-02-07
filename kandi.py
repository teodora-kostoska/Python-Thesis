#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:59:08 2020

@author: teodora
"""

import nltk 
import string 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import numpy as np
import sklearn.metrics as metrics 
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from nltk.corpus import wordnet 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from scipy.sparse import hstack
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#load data
fake = pd.read_csv('archive/Fake.csv')
real = pd.read_csv('archive/True.csv')

#create labels

real['label']=1
fake['label']=0

#combine data into one

data = pd.concat([real,fake])
data.head
data.shape

#shuffle data
data = data.sample(frac=1)

#lets perform basic analysis of data

#are there any empty values in the data set
#source rows 53-66: George 2020.
data.isnull().sum()
#no null values

#is data balanced
sns.countplot(data['label']);
#data appears to be balanced

#data distribution by subject and label
data['subject'].value_counts()
plt.figure(figsize = (10,10))
sns.countplot(data['subject'], color='blue');

plot=sns.countplot(x='label', hue='subject', data=data)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90)


#lets clean up data and preprocess it
#lets remove all unnecessary values in the data set i.e. title, subject, date
data = data.drop(['title', 'subject', 'date'], axis=1)
#source rows 73-88: Vazques, 2020.
#text to lowercase
data['text'] = data['text'].apply(lambda x: x.lower())

#remove punctuation
def remove_punct(text):
    word_list = [char for char in text if char not in string.punctuation]
    clean = ''.join(word_list)
    return clean

data['text'] = data['text'].apply(remove_punct)

#remove stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#tokenization
nltk.download('punkt')
data['text'] = data['text'].apply(lambda x: word_tokenize(x))


#lemmatization 
nltk.download('averaged_perceptron_tagger') 
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
#source rows 101-122: Prabhakaran, 2018.
def pos_tagger(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
def lemmatized_text(text):
    tagged = nltk.pos_tag(text)
    returnable = list(map(lambda x: (x[0], pos_tagger(x[1])), tagged))
    lemmatized_sentence = []
    for word, tag in returnable:
        if tag is None: 
            lemmatized_sentence.append(word)
        else: 
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_sentence

data['text'] = data['text'].apply(lambda x: lemmatized_text(x))


#split the data into training (80%) and testing (20%) data
X = data['text']
Y = data['label']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 5)

#TF-IDF
def tokenize_off(text):
    return text

vectorizer = TfidfVectorizer(tokenizer=tokenize_off, lowercase = False)

tfidf_train = vectorizer.fit_transform(x_train)

tfidf_test = vectorizer.transform(x_test)
    
#run control Naive Bayes code
NBclassifier = MultinomialNB()
NBclassifier.fit(tfidf_train, y_train)

predictionMNB = NBclassifier.predict(tfidf_test)

print(NBclassifier.score(tfidf_train, y_train))
#accuracy: 0.97372

print(NBclassifier.score(tfidf_test, y_test))
#accuracy 0.96915

#run control SVM code
SVM = SVC(kernel='linear')
SVM.fit(tfidf_train, y_train)

predictionSVM = SVM.predict(tfidf_test)

print(SVM.score(tfidf_train, y_train))
#precision: 0.99947
print(SVM.score(tfidf_test, y_test))
#precission: 0.99677


#evaluate performance

#Multinomial Naive Bayes
#source rows 171-190: George,2020.
score = metrics.accuracy_score(y_test, predictionMNB)
print("accuracy:   %0.3f" % (score*100))
cm = metrics.confusion_matrix(y_test, predictionMNB, labels=[0,1])

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, predictionMNB),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()

#Support Vector Machine
score = metrics.accuracy_score(y_test, predictionSVM)
print("accuracy:   %0.3f" % (score*100))
cm = metrics.confusion_matrix(y_test, predictionSVM, labels=[0,1])

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, predictionSVM),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()

#TEXTBLOB

#for the sentiment analysis we need unprocessed data
data_sentiment = pd.concat([real,fake])
data_sentiment = data_sentiment.sample(frac=1)
data_sentiment = data_sentiment.drop(['title', 'subject', 'date'], axis=1)

#lets perform sentiment analysis with Text Blob
data_sentiment['textblob'] = data_sentiment['text'].apply(lambda x: TextBlob(x).polarity)

def sentiment_transformation(num):
    if num <=-0.05:
        return 0
    elif num >=0.05:
        return 2
    else:
        return 1
data_sentiment['textblob'] = data_sentiment['textblob'].apply(lambda x: sentiment_transformation(x))

#preprocess text data
#source rows 213-220: Vazques, 2020.
#text to lowercase
data_sentiment['text'] = data_sentiment['text'].apply(lambda x: x.lower())

#remove punctuation
data_sentiment['text'] = data_sentiment['text'].apply(remove_punct)

#remove stopwords
data_sentiment['text'] = data_sentiment['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#tokenization
data_sentiment['text'] = data_sentiment['text'].apply(lambda x: word_tokenize(x))

#lemmatization 
data_sentiment['text'] = data_sentiment['text'].apply(lambda x: lemmatized_text(x))

#split data
XTB = np.asarray(data_sentiment[['text', 'textblob']])
YTB = data_sentiment['label']
x_trainTB, x_testTB, y_trainTB, y_testTB = train_test_split(XTB,YTB, test_size = 0.2, random_state = 5)

#TF-IDF
XTB_train = x_trainTB[:,0]
tfidf_trainTB = vectorizer.fit_transform(XTB_train)
XTB_test = x_testTB[:,0]
tfidf_testTB = vectorizer.transform(XTB_test)

#append the textblob results to tf idf matrix
XTB_train = x_trainTB[:,1]
tfidf_trainTB = hstack((tfidf_trainTB, np.array(XTB_train,dtype=float)[:,None]))
tfidf_trainTB.shape

#same for testing matrix
XTB_test = x_testTB[:,1]
tfidf_testTB = hstack((tfidf_testTB, np.array(XTB_test, dtype=float)[:,None]))
tfidf_testTB.shape

#lets run Naive bayes classifier w TextBlob
NBclassifier.fit(tfidf_trainTB, y_trainTB)

predictionMNBTB = NBclassifier.predict(tfidf_testTB)

print(NBclassifier.score(tfidf_trainTB, y_trainTB))
#precission: 0.974497
print(NBclassifier.score(tfidf_testTB, y_testTB))
#precision: 0.968597

#lets run SVM classifier w TextBlob
SVM.fit(tfidf_trainTB, y_trainTB)

predictionSVMTB = SVM.predict(tfidf_testTB)

print(SVM.score(tfidf_trainTB, y_trainTB))
#precision: 0.99961
print(SVM.score(tfidf_testTB, y_testTB))
#precision: 0.99699

#evaluate performance
#Multinomial Naive Bayes
#source rows 272-291: George, 2020.
score = metrics.accuracy_score(y_testTB, predictionMNBTB)
print("accuracy:   %0.3f" % (score*100))
cm = metrics.confusion_matrix(y_testTB, predictionMNBTB, labels=[0,1])

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_testTB, predictionMNBTB),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()

#Support Vector Machine
score = metrics.accuracy_score(y_testTB, predictionSVMTB)
print("accuracy:   %0.3f" % (score*100))
cm = metrics.confusion_matrix(y_testTB, predictionSVMTB, labels=[0,1])

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_testTB, predictionSVMTB),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()

#VADER

#for the sentiment analysis we need unprocessed data
data_sentimentV = pd.concat([real,fake])
data_sentimentV = data_sentimentV.sample(frac=1)
data_sentimentV = data_sentimentV.drop(['title', 'subject', 'date'], axis=1)

#lets perform sentiment analysis with Vader
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()
data_sentimentV['sentiment'] = data_sentimentV['text'].apply(lambda x: vader.polarity_scores(x)['compound'])
data_sentimentV['sentiment'] = data_sentimentV['sentiment'].apply(lambda x: sentiment_transformation(x))

#preprocess text data
#source rows 307-314: Vazques, 2020.
#text to lowercase
data_sentimentV['text'] = data_sentimentV['text'].apply(lambda x: x.lower())

#remove punctuation
data_sentimentV['text'] = data_sentimentV['text'].apply(remove_punct)

#remove stopwords
data_sentimentV['text'] = data_sentimentV['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#tokenization
data_sentimentV['text'] = data_sentimentV['text'].apply(lambda x: word_tokenize(x))

#lemmatization 
data_sentimentV['text'] = data_sentimentV['text'].apply(lambda x: lemmatized_text(x))

#split data
XV = np.asarray(data_sentimentV[['text', 'sentiment']])
YV = data_sentimentV['label']
x_trainV, x_testV, y_trainV, y_testV = train_test_split(XV,YV, test_size = 0.2, random_state = 5)

#TF-IDF
XV_train = x_trainV[:,0]
tfidf_trainV = vectorizer.fit_transform(XV_train)
XV_test = x_testV[:,0]
tfidf_testV = vectorizer.transform(XV_test)

#append the vader results to tf idf matrix
XV_train = x_trainV[:,1]
tfidf_trainV = hstack((tfidf_trainV, np.array(XV_train,dtype=float)[:,None]))
tfidf_trainV.shape

#same for testing matrix
XV_test = x_testV[:,1]
tfidf_testV = hstack((tfidf_testV, np.array(XV_test, dtype=float)[:,None]))
tfidf_testV.shape

#lets run Naive bayes classifier w Vader
NBclassifier.fit(tfidf_trainV, y_trainV)

predictionMNBV = NBclassifier.predict(tfidf_testV)

print(NBclassifier.score(tfidf_trainV, y_trainV))
#precission:0.95860
print(NBclassifier.score(tfidf_testV, y_testV))
#precision: 0.95033

#lets run SVM classifier w Vader
SVM.fit(tfidf_trainV, y_trainV)

predictionSVMV = SVM.predict(tfidf_testV)

print(SVM.score(tfidf_trainV, y_trainV))
#precission: 0.999499
print(SVM.score(tfidf_testV, y_testV))
#precission: 0.99655

#evaluate performance
#Multinomial Naive Bayes
#source rows 367-286: George 2020.
score = metrics.accuracy_score(y_testV, predictionMNBV)
print("accuracy:   %0.3f" % (score*100))
cm = metrics.confusion_matrix(y_testV, predictionMNBV, labels=[0,1])

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_testV, predictionMNBV),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()

#Support Vector Machine
score = metrics.accuracy_score(y_testV, predictionSVMV)
print("accuracy:   %0.3f" % (score*100))
cm = metrics.confusion_matrix(y_testV, predictionSVMV, labels=[0,1])

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_testV, predictionSVMV),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()