#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import bs4
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import re

# Import Dependencies & Load dataset
import requests
import spacy
from spacy import displacy
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span
from tqdm import tqdm

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('english'))


# In[2]:


# Import Dataset

data = pd.read_csv("task1_test_no_label.csv", header=0, usecols=[0,1,3])

# merge two columns of title and description to Content

data['Content'] = data['title'] +" "+ data['description'].astype(str)

data.shape


# In[3]:


# Knowledge Graph

def get_entities(sent):
  ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""
    
    #############################################################
    
    for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
      # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " "+ tok.text

      # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
            # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " "+ tok.text

      ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier +" "+ prefix + " "+ tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""      

            ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier +" "+ prefix +" "+ tok.text

    ## chunk 5  
    # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text

  #############################################################
    return [ent2.strip(), ent1.strip()]

entity_pairs = []

counter = 0
for i in tqdm(data['Content']):
    entity_pairs.append([])
    entity_pairs[counter].extend(get_entities(i))

    counter = counter + 1

data['Content'] = entity_pairs


# In[4]:


# Remove StopWords, words less 2 letter

def remove_stp_less2letter(text):
    words = [w for w in text if w not in stopwords.words('english')]
    words = " ". join([w for w in words if len(w)>2])
    return words

data['Content'] = data['Content'].apply(lambda x: remove_stp_less2letter(x))


# In[5]:


# Remove http, html, digits

def remove_http_html_digit_punc(text):
    
    soup = BeautifulSoup(text, 'html.parser')
    html_free = soup.get_text()
    no_http = re.sub(r"http\S+", '', html_free)
    no_digit= re.sub(r"[0-9]","",no_http)
    no_p = "". join([c for c in no_digit if c not in string.punctuation])
    return no_p

data['Content'] = data['Content'].apply(lambda x:remove_http_html_digit_punc(x))


# In[6]:


data.to_csv("Preprocess_Test_KG.csv")
# data.to_csv("Preprocess_Train_KG.csv")


# In[7]:


# Import Dataset

data = pd.read_csv("task1_test_no_label.csv", header=0, usecols=[0,1,3])

# merge two columns of title and description to Content

data['Content'] = data['title'] +" "+ data['description'].astype(str)

data.shape


# In[8]:


# Yake- Ky-Extraction

import yake

keywords_Yake = []

language = "en"
max_ngram_size = 2
deduplication_threshold = 0.9
numOfKeywords = 10

kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
for i in range(len(data['Content'])):
    keywords = kw_extractor.extract_keywords(data['Content'][i])
    keywords_Yake.append([])
    for kw in keywords:
        keywords_Yake[i].append(kw[0])
        

# remove none value in list of keywords
keywords_Yake_new = []
for val in keywords_Yake:
    if len(val) != 1 :
        keywords_Yake_new.append(val)

data['Content'] = keywords_Yake

data['Content']


# In[9]:


# Remove StopWords, words less 2 letter

def remove_stp_less2letter(text):
    words = [w for w in text if w not in stopwords.words('english')]
    words = " ". join([w for w in words if len(w)>2])
    return words

data['Content'] = data['Content'].apply(lambda x: remove_stp_less2letter(x))


# In[10]:


# Remove http, html, digits

def remove_http_html_digit_punc(text):
    
    soup = BeautifulSoup(text, 'html.parser')
    html_free = soup.get_text()
    no_http = re.sub(r"http\S+", '', html_free)
    no_digit= re.sub(r"[0-9]","",no_http)
    no_p = "". join([c for c in no_digit if c not in string.punctuation])
    return no_p

data['Content'] = data['Content'].apply(lambda x:remove_http_html_digit_punc(x))


# In[11]:


data.to_csv("Preprocess_Test_Yake.csv")
# data.to_csv("Preprocess_Train_Yake.csv")


# In[ ]:





# In[ ]:




