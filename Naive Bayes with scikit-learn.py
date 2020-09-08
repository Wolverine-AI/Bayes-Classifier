#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import recall_score, precision_score, f1_score


# In[2]:


DATA_JSON_FILE = 'SpamData/Processing/email-text-data.json'


# In[3]:


data = pd.read_json(DATA_JSON_FILE)


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.sort_index(inplace=True)


# In[7]:


data.tail()


# In[9]:


vectorizer = CountVectorizer(stop_words='english')


# In[10]:


all_features = vectorizer.fit_transform(data.MESSAGE)


# In[11]:


all_features.shape


# In[12]:


vectorizer.vocabulary_


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(all_features, data.CATEGORY, 
                                                   test_size=0.3, random_state=88)


# In[16]:


X_train.shape


# In[18]:


X_test.shape


# In[20]:


classifier = MultinomialNB()


# In[21]:


classifier.fit(X_train, y_train)


# In[24]:


nr_correct = (y_test == classifier.predict(X_test)).sum()


# In[25]:


print(f'{nr_correct} documents classfied correctly')


# In[26]:


nr_incorrect = y_test.size - nr_correct


# In[27]:


print(f'Number of documents incorrectly classified is {nr_incorrect}')


# In[29]:


fraction_wrong = nr_incorrect / (nr_correct + nr_incorrect)
print(f'The (testing) accuracy of the model is {1-fraction_wrong:.2%}')


# In[30]:


classifier.score(X_test, y_test)


# In[32]:


recall_score(y_test, classifier.predict(X_test))


# In[33]:


precision_score(y_test, classifier.predict(X_test))


# In[34]:


f1_score(y_test, classifier.predict(X_test))


# In[35]:


example = ['get viagra for free now!', 
          'need a mortgage? Reply to arrange a call with a specialist and get a quote', 
          'Could you please help me with the project for tomorrow?', 
          'Hello Jonathan, how about a game of golf tomorrow?', 
          'Ski jumping is a winter sport in which competitors aim to achieve the longest jump after descending from a specially designed ramp on their skis. Along with jump length, competitor\'s style and other factors affect the final score. Ski jumping was first contested in Norway in the late 19th century, and later spread through Europe and North America in the early 20th century. Along with cross-country skiing, it constitutes the traditional group of Nordic skiing disciplines.'
          ]


# In[36]:


doc_term_matrix = vectorizer.transform(example)


# In[37]:


classifier.predict(doc_term_matrix)


# In[ ]:




