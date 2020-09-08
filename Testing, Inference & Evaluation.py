#!/usr/bin/env python
# coding: utf-8

# # Notebook Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Constants

# In[1]:


TOKEN_SPAM_PROB_FILE = 'SpamData/Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = 'SpamData/Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = 'SpamData/Testing/prob-all-tokens.txt'

TEST_FEATURE_MATRIX = 'SpamData/Testing/test-features.txt'
TEST_TARGET_FILE = 'SpamData/Testing/test-target.txt'

VOCAB_SIZE = 2500


# # Load the Data

# In[3]:


# Features
X_test = np.loadtxt(TEST_FEATURE_MATRIX, delimiter=' ')
# Target
y_test = np.loadtxt(TEST_TARGET_FILE, delimiter=' ')
# Token Probabilities
prob_token_spam = np.loadtxt(TOKEN_SPAM_PROB_FILE, delimiter=' ')
prob_token_ham = np.loadtxt(TOKEN_HAM_PROB_FILE, delimiter=' ')
prob_all_tokens = np.loadtxt(TOKEN_ALL_PROB_FILE, delimiter=' ')


# In[4]:


X_test[:5]


# # Calculating the Joint Probability
# 
# ### The Dot Product

# In[5]:


a = np.array([1, 2, 3])
b = np.array([0, 5, 4])
print('a = ', a)
print('b = ', b)


# In[6]:


a.dot(b)


# In[7]:


1*0 + 2*5 + 3*4


# In[8]:


c = np.array([[0, 6], [3, 0], [5, 1]])
print('shape of c is', c.shape)
print(c)


# In[9]:


print(a.dot(c))
print('shape of the dot product is', a.dot(c).shape)


# In[11]:


X_test.shape


# In[12]:


prob_token_spam.shape


# In[13]:


print('shape of the dot product is ', X_test.dot(prob_token_spam).shape)


# ## Set the Prior
# 
# $$P(Spam \, | \, X) = \frac{P(X \, | \, Spam) \, P(Spam)} {P(X)}$$

# In[14]:


PROB_SPAM = 0.3116


# In[15]:


np.log(prob_token_spam)


# ## Joint probability in log format

# In[16]:


joint_log_spam = X_test.dot(np.log(prob_token_spam) - np.log(prob_all_tokens)) + np.log(PROB_SPAM)


# In[17]:


joint_log_spam[:5]


# $$P(Ham \, | \, X) = \frac{P(X \, | \, Ham) \, (1-P(Spam))} {P(X)}$$

# In[18]:


joint_log_ham = X_test.dot(np.log(prob_token_ham) - np.log(prob_all_tokens)) + np.log(1-PROB_SPAM)


# In[19]:


joint_log_ham[:5]


# In[20]:


joint_log_ham.size


# # Making Predictions
# 
# ### Checking for the higher joint probability
# 
# $$P(Spam \, | \, X) \, > \, P(Ham \, | \, X)$$
# <center>**OR**</center>
# <br>
# $$P(Spam \, | \, X) \, < \, P(Ham \, | \, X)$$

# In[21]:


prediction = joint_log_spam > joint_log_ham


# In[22]:


prediction[-5:]*1


# In[23]:


y_test[-5:]


# ### Simplify
# 
# $$P(X \, | \, Spam) \, P(Spam) ≠  \frac{P(X \, | \, Spam) \, P(Spam)}{P(X)}$$

# In[24]:


joint_log_spam = X_test.dot(np.log(prob_token_spam)) + np.log(PROB_SPAM)
joint_log_ham = X_test.dot(np.log(prob_token_ham)) + np.log(1-PROB_SPAM)


# # Metrics and Evaluation
# 
# ## Accuracy

# In[25]:


correct_docs = (y_test == prediction).sum()
print('Docs classified correctly', correct_docs)
numdocs_wrong = X_test.shape[0] - correct_docs
print('Docs classified incorrectly', numdocs_wrong)


# In[26]:


# Accuracy
correct_docs/len(X_test)


# In[27]:


fraction_wrong = numdocs_wrong/len(X_test)
print('Fraction classified incorrectly is {:.2%}'.format(fraction_wrong))
print('Accuracy of the model is {:.2%}'.format(1-fraction_wrong))


# ## Visualising the Results

# In[28]:


# Chart Styling Info
yaxis_label = 'P(X | Spam)'
xaxis_label = 'P(X | Nonspam)'

linedata = np.linspace(start=-14000, stop=1, num=1000)


# In[29]:


plt.figure(figsize=(11, 7))
plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)

# Set scale
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])

plt.scatter(joint_log_ham, joint_log_spam, color='navy')
plt.show()


# ## The Decision Boundary

# In[30]:


plt.figure(figsize=(11, 7))
plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)

# Set scale
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])

plt.scatter(joint_log_ham, joint_log_spam, color='navy', alpha=0.5, s=25)
plt.plot(linedata, linedata, color='orange')

plt.show()


# In[31]:


plt.figure(figsize=(16, 7))

# Chart Nr 1:
plt.subplot(1, 2, 1)

plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)

# Set scale
plt.xlim([-14000, 1])
plt.ylim([-14000, 1])

plt.scatter(joint_log_ham, joint_log_spam, color='navy', alpha=0.5, s=25)
plt.plot(linedata, linedata, color='orange')

# Chart Nr 2:
plt.subplot(1, 2, 2)

plt.xlabel(xaxis_label, fontsize=14)
plt.ylabel(yaxis_label, fontsize=14)

# Set scale
plt.xlim([-2000, 1])
plt.ylim([-2000, 1])

plt.scatter(joint_log_ham, joint_log_spam, color='navy', alpha=0.5, s=3)
plt.plot(linedata, linedata, color='orange')

plt.show()


# In[32]:


# Chart Styling
sns.set_style('whitegrid')
labels = 'Actual Category'

summary_df = pd.DataFrame({yaxis_label: joint_log_spam, xaxis_label: joint_log_ham, 
                          labels: y_test})


# In[33]:


sns.lmplot(x=xaxis_label, y=yaxis_label, data=summary_df, size=6.5, fit_reg=False,
          scatter_kws={'alpha': 0.5, 's': 25})

plt.xlim([-2000, 1])
plt.ylim([-2000, 1])

plt.plot(linedata, linedata, color='black')

sns.plt.show()


# In[34]:


sns.lmplot(x=xaxis_label, y=yaxis_label, data=summary_df, size=6.5, fit_reg=False, legend=False,
          scatter_kws={'alpha': 0.5, 's': 25}, hue=labels, markers=['o', 'x'], palette='hls')

plt.xlim([-2000, 1])
plt.ylim([-2000, 1])

plt.plot(linedata, linedata, color='black')

plt.legend(('Decision Boundary', 'Nonspam', 'Spam'), loc='lower right', fontsize=14)

sns.plt.show()


# In[35]:


my_colours = ['#4A71C0', '#AB3A2C']

sns.lmplot(x=xaxis_label, y=yaxis_label, data=summary_df, size=6.5, fit_reg=False, legend=False,
          scatter_kws={'alpha': 0.7, 's': 25}, hue=labels, markers=['o', 'x'], palette=my_colours)

plt.xlim([-500, 1])
plt.ylim([-500, 1])

plt.plot(linedata, linedata, color='black')

plt.legend(('Decision Boundary', 'Nonspam', 'Spam'), loc='lower right', fontsize=14)

sns.plt.show()


# ### False Positives and False Negatives

# In[36]:


np.unique(prediction, return_counts=True)


# In[37]:


true_pos = (y_test == 1) & (prediction == 1)


# In[38]:


true_pos.sum()


# In[39]:


false_pos = (y_test == 0) & (prediction == 1)
false_pos.sum()


# In[40]:


false_neg = (y_test == 1) & (prediction == 0)
false_neg.sum()


# ## Recall Score

# In[41]:


recall_score = true_pos.sum() / (true_pos.sum() + false_neg.sum())
print('Recall score is {:.2%}'.format(recall_score))


# ## Precision Score

# In[42]:


precision_score = true_pos.sum() / (true_pos.sum() + false_pos.sum())
print('Precision score is {:.3}'.format(precision_score))


# ## F-Score or F1 Score

# In[44]:


f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
print('F Score is {:.2}'.format(f1_score))


# In[ ]:




