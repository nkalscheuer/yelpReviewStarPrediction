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


# In[10]:


vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print("X train: \n{}".format(repr(X_train)))


# In[12]:


feature_names = vect.get_feature_names()
print("Number of features:{}".format(len(feature_names)))
print("First 50 features:\n{}".format(feature_names[:50]))
print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("Every 2000th feature:\n{}".format(feature_names[::2000]))


# In[17]:


import time
def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))
#input("Press Enter to start")
#start_time = time.time()
#input("Press Enter to stop")
#end_time = time.time()
#time_lapsed = end_time - start_time
#time_convert(time_lapsed)


# In[16]:


# Finally we're machine learning... haha
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
start_time = time.time()
print("Time start:{}".format(time.time()))
scores = cross_val_score(LogisticRegression(solver='liblinear', multi_class='auto', max_iter=150),X_train, y_train, cv=5)
end_time = time.time()
print("Mean cross validation accuracy: {:.2f}".format(np.mean(scores)))
time_lapsed = end_time - start_time
time_convert(time_lapsed)


# In[19]:


# Finally we're machine learning... haha
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.001, 0.01, 0.1, 1, 10]}
start_time = time.time()
print("Time start:{}".format(time.time()))
grid = GridSearchCV(LogisticRegression(solver='liblinear', multi_class='auto', max_iter=150), param_grid, cv=5)
grid.fit(X_train, y_train)
end_time = time.time()
print("Best cross validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: {}", grid.best_params_)
time_lapsed = end_time - start_time
time_convert(time_lapsed)


# In[ ]:




