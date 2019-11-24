#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().system('tree data\\aclImdb')


# In[2]:


from sklearn.datasets import load_files

reviews_train = load_files("data/aclImdb/train/")
# load files returns a bunch, containing training texts and training lables
text_train, y_train = reviews_train.data, reviews_train.target
print("type of text train: {}".format(type(text_train)))
print("length of text train: {}".format(len(text_train)))
print("text_train[1]:\n{}".format(text_train[1]))


# In[3]:


print("Hello world")


# In[4]:


text_train = [doc.replace(b"<br />", b" ") for doc in text_train]


# In[5]:


print("Samples per class(training): {}".format(np.bincount(y_train)))


# In[6]:


reviews_test = load_files("data/aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
print("type of text test: {}".format(type(text_test)))
print("length of text test: {}".format(len(text_test)))
print("text_test[1]:\n{}".format(text_test[1]))
print("Samples per class (test): {}".format(np.bincount(y_test)))
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]


# In[7]:


nicks_words = ["My name is Polly and Polly wants a cracker", "There are a lot of crackers in this establishment."]


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(nicks_words)
print("Vocabulary size: {}".format(len(vect.vocabulary_)))
print("Vocabulary content:\n {}".format(vect.vocabulary_))


# In[9]:


bag_of_words = vect.transform(nicks_words)
print("Dense representation of bag_of_words: {}".format(bag_of_words.toarray()))


# In[ ]:




