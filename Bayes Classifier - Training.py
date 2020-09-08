#!/usr/bin/env python
# coding: utf-8

# # Notebook Imports

# In[1]:


import pandas as pd
import numpy as np


# # Constants

# In[1]:


TRAINING_DATA_FILE = 'SpamData/Training/train-data.txt'
TEST_DATA_FILE = 'SpamData/Training/test-data.txt'

TOKEN_SPAM_PROB_FILE = 'SpamData/Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = 'SpamData/Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = 'SpamData/Testing/prob-all-tokens.txt'

TEST_FEATURE_MATRIX = 'SpamData/Testing/test-features.txt'
TEST_TARGET_FILE = 'SpamData/Testing/test-target.txt'

VOCAB_SIZE = 2500


# # Read and Load Features from .txt Files into NumPy Array

# In[3]:


sparse_train_data = np.loadtxt(TRAINING_DATA_FILE, delimiter=' ', dtype=int)


# In[4]:


sparse_test_data = np.loadtxt(TEST_DATA_FILE, delimiter=' ', dtype=int)


# In[5]:


sparse_train_data[:5]


# In[6]:


sparse_train_data[-5:]


# In[7]:


print('Nr of rows in training file', sparse_train_data.shape[0])
print('Nr of rows in test file', sparse_test_data.shape[0])


# In[8]:


print('Nr of emails in training file', np.unique(sparse_train_data[:, 0]).size)


# In[9]:


print('Nr of emails in test file', np.unique(sparse_test_data[:, 0]).size)


# ### How to create an Empty DataFrame

# In[10]:


column_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, VOCAB_SIZE))
column_names[:5]


# In[11]:


len(column_names)


# In[12]:


index_names = np.unique(sparse_train_data[:, 0])
index_names


# In[13]:


full_train_data = pd.DataFrame(index=index_names, columns=column_names)
full_train_data.fillna(value=0, inplace=True)


# In[14]:


full_train_data.head()


# In[15]:


sparse_train_data[10:13]


# In[16]:


sparse_train_data[10][3]


# # Create a Full Matrix from a Sparse Matrix

# In[66]:


def make_full_matrix(sparse_matrix, nr_words, doc_idx=0, word_idx=1, cat_idx=2, freq_idx=3):
    """
    Form a full matrix from a sparse matrix. Return a pandas dataframe. 
    Keyword arguments:
    sparse_matrix -- numpy array
    nr_words -- size of the vocabulary. Total number of tokens. 
    doc_idx -- position of the document id in the sparse matrix. Default: 1st column
    word_idx -- position of the word id in the sparse matrix. Default: 2nd column
    cat_idx -- position of the label (spam is 1, nonspam is 0). Default: 3rd column
    freq_idx -- position of occurrence of word in sparse matrix. Default: 4th column
    """
    column_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, VOCAB_SIZE))
    doc_id_names = np.unique(sparse_matrix[:, 0])
    full_matrix = pd.DataFrame(index=doc_id_names, columns=column_names)
    full_matrix.fillna(value=0, inplace=True)
    
    for i in range(sparse_matrix.shape[0]):
        doc_nr = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        label = sparse_matrix[i][cat_idx]
        occurrence = sparse_matrix[i][freq_idx]
        
        full_matrix.at[doc_nr, 'DOC_ID'] = doc_nr
        full_matrix.at[doc_nr, 'CATEGORY'] = label
        full_matrix.at[doc_nr, word_id] = occurrence
    
    full_matrix.set_index('DOC_ID', inplace=True)
    return full_matrix
    


# In[18]:


get_ipython().run_cell_magic('time', '', 'full_train_data = make_full_matrix(sparse_train_data, VOCAB_SIZE)')


# In[19]:


full_train_data.tail()


# # Training the Naive Bayes Model
# 
# ## Calculating the Probability of Spam

# In[21]:


full_train_data.CATEGORY.size


# In[22]:


full_train_data.CATEGORY.sum()


# In[23]:


prob_spam = full_train_data.CATEGORY.sum() / full_train_data.CATEGORY.size
print('Probability of spam is', prob_spam)


# ## Total Number of Words / Tokens

# In[24]:


full_train_features = full_train_data.loc[:, full_train_data.columns != 'CATEGORY']
full_train_features.head()


# In[25]:


email_lengths = full_train_features.sum(axis=1)
email_lengths.shape


# In[26]:


email_lengths[:5]


# In[27]:


total_wc = email_lengths.sum()
total_wc


# ## Number of Tokens in Spam & Ham Emails

# In[28]:


spam_lengths = email_lengths[full_train_data.CATEGORY == 1]
spam_lengths.shape


# In[29]:


spam_wc = spam_lengths.sum()
spam_wc


# In[30]:


ham_lengths = email_lengths[full_train_data.CATEGORY == 0]
ham_lengths.shape


# In[31]:


email_lengths.shape[0] - spam_lengths.shape[0] - ham_lengths.shape[0]


# In[32]:


nonspam_wc = ham_lengths.sum()
nonspam_wc


# In[33]:


spam_wc + nonspam_wc - total_wc


# In[35]:


print('Average nr of words in spam emails {:.0f}'.format(spam_wc / spam_lengths.shape[0]))
print('Average nr of words in ham emails {:.3f}'.format(nonspam_wc / ham_lengths.shape[0]))


# ## Summing the Tokens Occuring in Spam

# In[36]:


full_train_features.shape


# In[37]:


train_spam_tokens = full_train_features.loc[full_train_data.CATEGORY == 1]
train_spam_tokens.head()


# In[38]:


train_spam_tokens.tail()


# In[39]:


train_spam_tokens.shape


# In[40]:


summed_spam_tokens = train_spam_tokens.sum(axis=0) + 1
summed_spam_tokens.shape


# In[41]:


summed_spam_tokens.tail()


# ## Summing the Tokens Occuring in Ham

# In[43]:


train_ham_tokens = full_train_features.loc[full_train_data.CATEGORY == 0]
summed_ham_tokens = train_ham_tokens.sum(axis=0) + 1


# In[44]:


summed_ham_tokens.shape


# In[45]:


summed_ham_tokens.tail()


# In[46]:


train_ham_tokens[2499].sum() + 1


# ## P(Token | Spam) - Probability that a Token Occurs given the Email is Spam

# In[47]:


prob_tokens_spam = summed_spam_tokens / (spam_wc + VOCAB_SIZE)
prob_tokens_spam[:5]


# In[48]:


prob_tokens_spam.sum()


# ## P(Token | Ham) - Probability that a Token Occurs given the Email is Nonspam

# In[49]:


prob_tokens_nonspam = summed_ham_tokens / (nonspam_wc + VOCAB_SIZE)
prob_tokens_nonspam.sum()


# # P(Token) - Probability that Token Occurs 

# In[50]:


prob_tokens_all = full_train_features.sum(axis=0) / total_wc


# In[51]:


prob_tokens_all.sum()


# # Save the Trained Model

# In[52]:


np.savetxt(TOKEN_SPAM_PROB_FILE, prob_tokens_spam)
np.savetxt(TOKEN_HAM_PROB_FILE, prob_tokens_nonspam)
np.savetxt(TOKEN_ALL_PROB_FILE, prob_tokens_all)


# # Prepare Test Data

# In[61]:


sparse_test_data.shape


# In[62]:


get_ipython().run_cell_magic('time', '', 'full_test_data = make_full_matrix(sparse_test_data, nr_words=VOCAB_SIZE)')


# In[63]:


X_test = full_test_data.loc[:, full_test_data.columns != 'CATEGORY']
y_test = full_test_data.CATEGORY


# In[65]:


np.savetxt(TEST_TARGET_FILE, y_test)
np.savetxt(TEST_FEATURE_MATRIX, X_test)


# In[ ]:




