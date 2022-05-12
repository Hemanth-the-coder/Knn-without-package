#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris


# In[2]:


iris = load_iris()
iris


# In[3]:


df=pd.DataFrame(iris.data)


# In[4]:


df.rename(columns = {0:'sepal length' , 1:'sepal width',2:'petal length' ,3:'petal width'  } , inplace = True)


# In[5]:


df


# In[6]:


df['target'] = iris.target


# In[7]:


df


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


y = np.asarray(df['target'])


# In[10]:


y


# In[11]:


df = df.drop('target', axis =1)


# In[12]:


df


# In[13]:


xtrain , xtest , ytrain , ytest = train_test_split(df , y , test_size = 0.3  , random_state = 4)


# In[14]:


xtrain 


# In[15]:


xtest


# In[16]:


ytrain 


# In[17]:


ytest


# In[18]:


def knnalgorithm(xtrain , k , arr , ytest ):
    distances = []
    results =[]
    for i in range(xtrain.shape[0]):
        distances.append(euclideandist(xtrain, i,arr))
    distances.sort()
    for i in distances[:5]:
        results.append(i[1])
    final = [ytest[i] for i in results]
    return max(final , key = final.count)
        
        
        
def euclideandist(xtrain ,i,arr):
    return [np.sqrt(np.sum((xtrain.iloc[i].values-arr)**2)) , i ]
        
    


# In[33]:


predicts =[]
for i in range(xtest.shape[0]):
    predicts.append(knnalgorithm(xtrain , 3, xtest.iloc[i].values , ytrain ))


# In[34]:


print(predicts)


# In[21]:


ytest


# In[22]:


#function to find the accuracy of one vs one classifier 
def accuracy(predicts):
    correct_predictions =0 
    for i in range(len(predicts)):
        if predicts[i]==ytest[i]:
            correct_predictions+=1
    return correct_predictions/len(predicts)


# In[32]:


accuracy(predicts)

