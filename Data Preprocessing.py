#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import re


# # Import Dataset

# In[2]:


# Import Dataset

data = pd.read_csv("task1_test_no_label.csv", header=0, usecols=[1,3])

# merge two columns of title and description to Content

data['Content'] = data['title'] +" "+ data['description'].astype(str)

data.shape


# # Remove HTML

# In[4]:


def remove_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    html_free = soup.get_text()
    return html_free


# In[6]:


data['Content'] = data['Content'].apply(lambda x:remove_html(x))


# # Remove http

# In[7]:


def remove_http(text):
    no_http = re.sub(r"http\S+", '', text)
    return no_http


# In[8]:


data['Content'] = data['Content'].apply(lambda x:remove_http(x))


# # Remove punctuation:

# In[9]:


def remove_punc(text):
    no_p = "". join([c for c in text if c not in string.punctuation])
    return no_p


# In[10]:


data['Content'] = data['Content'].apply(lambda x:remove_punc(x))


# # Remove Digits

# In[11]:


def remove_digit(text):
    re_digit = re.sub(r"\w*\d\w*", ' ', text)
    return re_digit


# In[12]:


data['Content'] = data['Content'].apply(lambda x:remove_digit(x))


# # Tokenize:

# In[13]:


# Instantiate Tokenizer
tokenizer = RegexpTokenizer(r'\w+')


# ### converting all letters to lower case

# In[14]:


data['Content'] = data['Content'].apply(lambda x:tokenizer.tokenize(x.lower()))


# # Remove words that length is less than 3

# In[15]:


def remove_less3word(text):
    words = [w for w in text if len(w)>=3]
    return words


# In[16]:


data['Content'] = data['Content'].apply(lambda x: remove_less3word(x))


# # Lemmatizing:

# In[17]:


# Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()


# In[18]:


def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text


# In[19]:


data['Content'] = data['Content'].apply(lambda x: word_lemmatizer(x))


# # Stemming

# In[20]:


# Instantiate Stemmer
stemmer = PorterStemmer()


# In[21]:


def word_stemmer(text):
    stem_text = " ". join([stemmer.stem(i) for i in text])
    return stem_text


# In[22]:


data['Content'] = data['Content'].apply(lambda x: word_stemmer(x))


# # Correcting

# In[ ]:


from textblob import TextBlob

data['Content'] = data['Content'].apply(lambda x: str(TextBlob(x).correct()))


# In[ ]:


# data.to_csv("Preprocess_Train.csv")
data.to_csv("Preprocess_Test.csv")


# In[ ]:





# In[ ]:




