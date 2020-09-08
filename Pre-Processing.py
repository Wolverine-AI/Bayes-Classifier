#!/usr/bin/env python
# coding: utf-8

# # Notebook Imports

# In[1]:


from os import walk
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup
from wordcloud import WordCloud
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# # Constants

# In[2]:


EXAMPLE_FILE = 'SpamData/Processing/practice_email.txt'

SPAM_1_PATH = 'SpamData/Processing/spam_assassin_corpus/spam_1'
SPAM_2_PATH = 'SpamData/Processing/spam_assassin_corpus/spam_2'
EASY_NONSPAM_1_PATH = 'SpamData/Processing/spam_assassin_corpus/easy_ham_1'
EASY_NONSPAM_2_PATH = 'SpamData/Processing/spam_assassin_corpus/easy_ham_2'

SPAM_CAT = 1
HAM_CAT = 0
VOCAB_SIZE = 2500

DATA_JSON_FILE = 'SpamData/Processing/email-text-data.json'
WORD_ID_FILE = 'SpamData/Processing/word-by-id.csv'

TRAINING_DATA_FILE = 'SpamData/Training/train-data.txt'
TEST_DATA_FILE = 'SpamData/Training/test-data.txt'

WHALE_FILE = 'SpamData/Processing/wordcloud_resources/whale-icon.png'
SKULL_FILE = 'SpamData/Processing/wordcloud_resources/skull-icon.png'
THUMBS_UP_FILE = 'SpamData/Processing/wordcloud_resources/thumbs-up.png'
THUMBS_DOWN_FILE = 'SpamData/Processing/wordcloud_resources/thumbs-down.png'
CUSTOM_FONT_FILE = 'SpamData/Processing/wordcloud_resources/OpenSansCondensed-Bold.ttf'


# # Reading Files

# In[4]:


stream = open(EXAMPLE_FILE, encoding='latin-1')
message = stream.read()
stream.close()

print(type(message))
print(message)


# In[5]:


import sys
sys.getfilesystemencoding()


# In[6]:


stream = open(EXAMPLE_FILE, encoding='latin-1')

is_body = False
lines = []

for line in stream:
    if is_body:
        lines.append(line)
    elif line == '\n':
        is_body = True

stream.close()

email_body = '\n'.join(lines)
print(email_body)


# # Generator Functions

# In[7]:


def generate_squares(N):
    for my_number in range(N):
        yield my_number ** 2


# In[8]:


for i in generate_squares(5):
    print(i, end=' ->')


# ## Email body extraction

# In[9]:


def email_body_generator(path):
    
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            
            filepath = join(root, file_name)
            
            stream = open(filepath, encoding='latin-1')

            is_body = False
            lines = []

            for line in stream:
                if is_body:
                    lines.append(line)
                elif line == '\n':
                    is_body = True

            stream.close()

            email_body = '\n'.join(lines)
            
            yield file_name, email_body


# In[10]:


def df_from_directory(path, classification):
    rows = []
    row_names = []
    
    for file_name, email_body in email_body_generator(path):
        rows.append({'MESSAGE': email_body, 'CATEGORY': classification})
        row_names.append(file_name)
        
    return pd.DataFrame(rows, index=row_names)


# In[11]:


spam_emails = df_from_directory(SPAM_1_PATH, 1)
spam_emails = spam_emails.append(df_from_directory(SPAM_2_PATH, 1))
spam_emails.head()


# In[12]:


spam_emails.shape


# In[13]:


ham_emails = df_from_directory(EASY_NONSPAM_1_PATH, HAM_CAT)
ham_emails = ham_emails.append(df_from_directory(EASY_NONSPAM_2_PATH, HAM_CAT))
ham_emails.shape


# In[14]:


data = pd.concat([spam_emails, ham_emails])
print('Shape of entire dataframe is ', data.shape)
data.head()


# In[15]:


data.tail()


# # Data Cleaning: Checking for Missing Values

# In[16]:


# checking if any message bodies are null
data['MESSAGE'].isnull().values.any()


# In[17]:


type("")


# In[18]:


len("")


# In[19]:


my_var = None


# In[20]:


type(my_var)


# In[21]:


# checking if there are empty emails (string length zero)
(data.MESSAGE.str.len() == 0).any()


# In[22]:


(data.MESSAGE.str.len() == 0).sum()


# In[23]:


data.MESSAGE.isnull().sum()


# ### Locate empty emails

# In[24]:


type(data.MESSAGE.str.len() == 0)


# In[25]:


data[data.MESSAGE.str.len() == 0].index


# In[26]:


data.index.get_loc('.DS_Store')


# In[27]:


data[4608:4611]


# # Remove System File Entries from Dataframe

# In[28]:


data.drop(['cmds', '.DS_Store'], inplace=True)

data[4608:4611]


# In[29]:


data.shape


# # Add Document IDs to Track Emails in Dataset

# In[30]:


document_ids = range(0, len(data.index))
data['DOC_ID'] = document_ids


# In[31]:


data['FILE_NAME'] = data.index
data.set_index('DOC_ID', inplace=True)
data.head()


# In[32]:


data.tail()


# # Save to File using Pandas

# In[33]:


data.to_json(DATA_JSON_FILE)


# # Number of Spam Messages Visualised (Pie Charts)

# In[34]:


data.CATEGORY.value_counts()


# In[35]:


amount_of_spam = data.CATEGORY.value_counts()[1]
amount_of_ham = data.CATEGORY.value_counts()[0]


# In[36]:


category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90, 
       autopct='%1.0f%%')
plt.show()


# In[37]:


category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]
custom_colours = ['#ff7675', '#74b9ff']

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90, 
       autopct='%1.0f%%', colors=custom_colours, explode=[0, 0.1])
plt.show()


# In[38]:


category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]
custom_colours = ['#ff7675', '#74b9ff']

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90, 
       autopct='%1.0f%%', colors=custom_colours, pctdistance=0.8)

# draw circle
centre_circle = plt.Circle((0, 0), radius=0.6, fc='white')
plt.gca().add_artist(centre_circle)

plt.show()


# In[39]:


category_names = ['Spam', 'Legit Mail', 'Updates', 'Promotions']
sizes = [25, 43, 19, 22]
custom_colours = ['#ff7675', '#74b9ff', '#55efc4', '#ffeaa7']
offset = [0.05, 0.05, 0.05, 0.05]

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90, 
       autopct='%1.0f%%', colors=custom_colours, pctdistance=0.8, explode=offset)

# draw circle
centre_circle = plt.Circle((0, 0), radius=0.6, fc='white')
plt.gca().add_artist(centre_circle)

plt.show()


# # Natural Language Processing

# ### Text Pre-Processing

# In[40]:


# convert to lower case
msg = 'All work and no play makes Jack a dull boy.'
msg.lower()


# ### Download the NLTK Resources (Tokenizer & Stopwords)

# In[41]:


nltk.download('punkt')


# In[42]:


nltk.download('stopwords')


# In[43]:


nltk.download('gutenberg')
nltk.download('shakespeare')


# ## Tokenising

# In[44]:


msg = 'All work and no play makes Jack a dull boy.'
word_tokenize(msg.lower())


# ## Removing Stop Words 

# In[45]:


stop_words = set(stopwords.words('english'))


# In[46]:


type(stop_words)


# In[47]:


if 'this' in stop_words: print('Found it!')


# In[48]:


# print out 'Nope. Not in here' if the word "hello" is not contained in stop_words


# In[49]:


if 'hello' not in stop_words: print('Nope. Not in here')


# In[50]:


msg = 'All work and no play makes Jack a dull boy. To be or not to be.'
words = word_tokenize(msg.lower())

filtered_words = []
# append non-stop words to filtered_words
for word in words:
    if word not in stop_words:
        filtered_words.append(word)

print(filtered_words)


# ## Word Stems and Stemming

# In[51]:


msg = 'All work and no play makes Jack a dull boy. To be or not to be.       Nobody expects the Spanish Inquisition!'
words = word_tokenize(msg.lower())

# stemmer = PorterStemmer()
stemmer = SnowballStemmer('english')


filtered_words = []
#append non-stop words to filtered_words
for word in words:
    if word not in stop_words:
        stemmed_word = stemmer.stem(word)
        filtered_words.append(stemmed_word)

print(filtered_words)


# ## Removing Punctuation

# In[52]:


'p'.isalpha()


# In[53]:


'?'.isalpha()


# In[54]:


msg = 'All work and no play makes Jack a dull boy. To be or not to be. ???       Nobody expects the Spanish Inquisition!'

words = word_tokenize(msg.lower())
stemmer = SnowballStemmer('english')
filtered_words = []

for word in words:
    if word not in stop_words and word.isalpha():
        stemmed_word = stemmer.stem(word)
        filtered_words.append(stemmed_word)

print(filtered_words)


# ## Removing HTML tags from Emails

# In[55]:


soup = BeautifulSoup(data.at[2, 'MESSAGE'], 'html.parser')
print(soup.prettify())


# In[56]:


soup.get_text()


# ## Functions for Email Processing

# In[57]:


def clean_message(message, stemmer=PorterStemmer(), 
                 stop_words=set(stopwords.words('english'))):
    
    # Converts to Lower Case and splits up the words
    words = word_tokenize(message.lower())
    
    filtered_words = []
    
    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
    
    return filtered_words


# In[58]:


clean_message(email_body)


# In[59]:


# Modify function to remove HTML tags. Then test on Email with DOC_ID 2. 
def clean_msg_no_html(message, stemmer=PorterStemmer(), 
                 stop_words=set(stopwords.words('english'))):
    
    # Remove HTML tags
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()
    
    # Converts to Lower Case and splits up the words
    words = word_tokenize(cleaned_text.lower())
    
    filtered_words = []
    
    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
#             filtered_words.append(word) 
    
    return filtered_words


# In[60]:


clean_msg_no_html(data.at[2, 'MESSAGE'])


# # Apply Cleaning and Tokenisation to all messages

# ### Slicing Dataframes and Series & Creating Subsets

# In[61]:


data.iat[2, 2]


# In[62]:


data.iloc[5:11]


# In[63]:


first_emails = data.MESSAGE.iloc[0:3]

nested_list = first_emails.apply(clean_message)


# In[64]:


# flat_list = []
# for sublist in nested_list:
#     for item in sublist:
#         flat_list.append(item)

flat_list = [item for sublist in nested_list for item in sublist]
        
len(flat_list)


# In[65]:


flat_list


# In[66]:


get_ipython().run_cell_magic('time', '', '\nnested_list = data.MESSAGE.apply(clean_msg_no_html)')


# In[67]:


nested_list.tail()


# ### Using Logic to Slice Dataframes

# In[68]:


data[data.CATEGORY == 1].shape


# In[69]:


data[data.CATEGORY == 1].tail()


# In[70]:


# creating two variables (doc_ids_spam, doc_ids_ham) which 
# hold onto the indices for the spam and the non-spam emails respectively. 


# In[71]:


doc_ids_spam = data[data.CATEGORY == 1].index
doc_ids_ham = data[data.CATEGORY == 0].index


# In[72]:


doc_ids_ham


# ### Subsetting a Series with an Index

# In[73]:


type(doc_ids_ham)


# In[74]:


type(nested_list)


# In[75]:


nested_list_ham = nested_list.loc[doc_ids_ham]


# In[76]:


nested_list_ham.shape


# In[77]:


nested_list_ham.tail()


# In[78]:


nested_list_spam = nested_list.loc[doc_ids_spam]


# In[80]:


flat_list_ham = [item for sublist in nested_list_ham for item in sublist]
normal_words = pd.Series(flat_list_ham).value_counts()

normal_words.shape[0] # total number of unique words in the non-spam messages


# In[81]:


normal_words[:10]


# In[82]:


flat_list_spam = [item for sublist in nested_list_spam for item in sublist]
spammy_words = pd.Series(flat_list_spam).value_counts()

spammy_words.shape[0] # total number of unique words in the spam messages


# In[83]:


spammy_words[:10]


# # Creating a Word Cloud

# In[84]:


word_cloud = WordCloud().generate(email_body)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[85]:


example_corpus = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
len(example_corpus)


# In[86]:


type(example_corpus)


# In[87]:


example_corpus


# In[88]:


word_list = [''.join(word) for word in example_corpus]
novel_as_string = ' '.join(word_list)


# In[89]:


icon = Image.open(WHALE_FILE)
image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
image_mask.paste(icon, box=icon)

rgb_array = np.array(image_mask) # converts the image object to an array

word_cloud = WordCloud(mask=rgb_array, background_color='white', 
                      max_words=400, colormap='ocean')

word_cloud.generate(novel_as_string)

plt.figure(figsize=[16, 8])
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[90]:


rgb_array.shape


# In[91]:


rgb_array[1023, 2047]


# In[92]:


rgb_array[500, 1000]


# In[94]:


hamlet_corpus = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
word_list = [''.join(word) for word in hamlet_corpus]
hamlet_as_string = ' '.join(word_list)

skull_icon = Image.open(SKULL_FILE)
image_mask = Image.new(mode='RGB', size=skull_icon.size, color=(255, 255, 255))
image_mask.paste(skull_icon, box=skull_icon)
rgb_array = np.array(image_mask)

word_cloud = WordCloud(mask=rgb_array, background_color='white',
                      colormap='bone', max_words=600)

word_cloud.generate(hamlet_as_string)

plt.figure(figsize=[16, 8])
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Word Cloud of Ham and Spam Messages

# In[95]:


icon = Image.open(THUMBS_UP_FILE)
image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
image_mask.paste(icon, box=icon)

rgb_array = np.array(image_mask) # converts the image object to an array

# Generate the text as a string for the word cloud
ham_str = ' '.join(flat_list_ham)

word_cloud = WordCloud(mask=rgb_array, background_color='white', 
                      max_words=500, colormap='winter')

word_cloud.generate(ham_str)

plt.figure(figsize=[16, 8])
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[97]:


icon = Image.open(THUMBS_DOWN_FILE)
image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
image_mask.paste(icon, box=icon)

rgb_array = np.array(image_mask) # converts the image object to an array

# Generate the text as a string for the word cloud
spam_str = ' '.join(flat_list_spam)

word_cloud = WordCloud(mask=rgb_array, background_color='white', max_font_size=300,
                      max_words=2000, colormap='gist_heat', font_path=CUSTOM_FONT_FILE)

word_cloud.generate(spam_str.upper())

plt.figure(figsize=[16, 8])
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# # Generate Vocabulary & Dictionary

# In[98]:


stemmed_nested_list = data.MESSAGE.apply(clean_msg_no_html)
flat_stemmed_list = [item for sublist in stemmed_nested_list for item in sublist]


# In[99]:


unique_words = pd.Series(flat_stemmed_list).value_counts()
print('Nr of unique words', unique_words.shape[0])
unique_words.head()


# In[100]:


# Creating subset of the series called 'frequent_words' that only contains
# the most common 2,500 words out of the total. Print out the top 10 words


# In[101]:


frequent_words = unique_words[0:VOCAB_SIZE]
print('Most common words: \n', frequent_words[:10])


# ## Create Vocabulary DataFrame with a WORD_ID

# In[102]:


word_ids = list(range(0, VOCAB_SIZE))
vocab = pd.DataFrame({'VOCAB_WORD': frequent_words.index.values}, index=word_ids)
vocab.index.name = 'WORD_ID'
vocab.head()


# In[ ]:





# ## Save the Vocabulary as a CSV File

# In[103]:


vocab.to_csv(WORD_ID_FILE, index_label=vocab.index.name, header=vocab.VOCAB_WORD.name)


# # Checking if a Word is Part of the Vocabulary

# In[105]:


any(vocab.VOCAB_WORD == 'machine') # inefficient


# In[106]:


'brew' in set(vocab.VOCAB_WORD) # better way


# # Find the Email with the Most Number of Words

# In[108]:


# For loop
clean_email_lengths = []
for sublist in stemmed_nested_list:
    clean_email_lengths.append(len(sublist))


# In[109]:


# Python List Comprehension
clean_email_lengths = [len(sublist) for sublist in stemmed_nested_list]
print('Nr words in the longest email:', max(clean_email_lengths))


# In[110]:


print('Email position in the list (and the data dataframe)', np.argmax(clean_email_lengths))


# In[111]:


stemmed_nested_list[np.argmax(clean_email_lengths)]


# In[112]:


data.at[np.argmax(clean_email_lengths), 'MESSAGE']


# # Generate Features & a Sparse Matrix
# 
# ### Creating a DataFrame with one Word per Column

# In[113]:


type(stemmed_nested_list)


# In[114]:


type(stemmed_nested_list.tolist())


# In[115]:


word_columns_df = pd.DataFrame.from_records(stemmed_nested_list.tolist())
word_columns_df.head()


# In[116]:


word_columns_df.shape


# ### Splitting the Data into a Training and Testing Dataset

# In[118]:


X_train, X_test, y_train, y_test = train_test_split(word_columns_df, data.CATEGORY,
                                                   test_size=0.3, random_state=42)


# In[119]:


print('Nr of training samples', X_train.shape[0])
print('Fraction of training set', X_train.shape[0] / word_columns_df.shape[0])


# In[120]:


X_train.index.name = X_test.index.name = 'DOC_ID'
X_train.head()


# In[121]:


y_train.head()


# ### Create a Sparse Matrix for the Training Data

# In[307]:


word_index = pd.Index(vocab.VOCAB_WORD)
type(word_index[3])


# In[308]:


word_index.get_loc('thu')


# In[124]:


def make_sparse_matrix(df, indexed_words, labels):
    """
    Returns sparse matrix as dataframe.
    
    df: A dataframe with words in the columns with a document id as an index (X_train or X_test)
    indexed_words: index of words ordered by word id
    labels: category as a series (y_train or y_test)
    """
    
    nr_rows = df.shape[0]
    nr_cols = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []
    
    for i in range(nr_rows):
        for j in range(nr_cols):
            
            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels.at[doc_id]
                
                item = {'LABEL': category, 'DOC_ID': doc_id,
                       'OCCURENCE': 1, 'WORD_ID': word_id}
                
                dict_list.append(item)
    
    return pd.DataFrame(dict_list)


# In[125]:


get_ipython().run_cell_magic('time', '', 'sparse_train_df = make_sparse_matrix(X_train, word_index, y_train)')


# In[126]:


sparse_train_df[:5]


# In[127]:


sparse_train_df.shape


# In[128]:


sparse_train_df[-5:]


# ### Combine Occurrences with the Pandas groupby() Method

# In[129]:


train_grouped = sparse_train_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()
train_grouped.head()


# In[130]:


vocab.at[0, 'VOCAB_WORD']


# In[131]:


data.MESSAGE[0]


# In[132]:


train_grouped = train_grouped.reset_index()
train_grouped.head()


# In[133]:


train_grouped.tail()


# In[134]:


vocab.at[1923, 'VOCAB_WORD']


# In[135]:


data.MESSAGE[5795]


# In[136]:


train_grouped.shape


# ### Save Training Data as .txt File

# In[137]:


np.savetxt(TRAINING_DATA_FILE, train_grouped, fmt='%d')


# In[138]:


train_grouped.columns


# In[139]:


X_test.head()


# In[140]:


y_test.head()


# In[141]:


X_test.shape


# In[142]:


get_ipython().run_cell_magic('time', '', 'sparse_test_df = make_sparse_matrix(X_test, word_index, y_test)')


# In[143]:


sparse_test_df.shape


# In[144]:


test_grouped = sparse_test_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum().reset_index()
test_grouped.head()


# In[312]:


test_grouped.shape


# In[311]:


np.savetxt(TEST_DATA_FILE, test_grouped, fmt='%d')


# # Pre-Processing Subtleties 

# We started with 5796 emails. We split it into 4057 emails for training and 1739 emails for testing. 
# 
# * Checking :How many individual emails were included in the testing .txt file? 
# * Counting the number in the test_grouped DataFrame.
# * After splitting and shuffling our data, how many emails were included in the X_test DataFrame?
# * Is the number the same? If not, which emails were excluded and why?
# * Lets Compare the DOC_ID values to find out.

# In[313]:


train_doc_ids = set(train_grouped.DOC_ID)
test_doc_ids = set(test_grouped.DOC_ID)


# In[316]:


len(test_doc_ids)


# In[318]:


len(X_test)


# In[321]:


set(X_test.index.values) - test_doc_ids # Excluded emails after pre-processing


# In[326]:


data.MESSAGE[14]


# In[327]:


data.loc[14]


# In[329]:


clean_msg_no_html(data.at[14, 'MESSAGE'])


# In[331]:


data.MESSAGE[1096]


# In[332]:


clean_msg_no_html(data.at[1096, 'MESSAGE'])


# In[334]:


clean_message(data.at[1096, 'MESSAGE'])


# In[ ]:




