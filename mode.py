#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearn.model_selection  import train_test_split


# In[ ]:


np.random.seed(1)


# In[2]:


df= pd.read_csv('breast-cancer.csv')


# In[3]:


# Creating a folder if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')


# # Pre-processing and Visualizing data

# In[4]:


def process_data(df):
    # Counting how many diagnosis are malignant(M) and how many are benign(B)
    diagnosis_all = list(df.shape)[0]
    diagnosis_categories = list(df['diagnosis'].value_counts())

    print("\n \t The data has {} diagnosis, {} malignant and {} benign.".format
          (diagnosis_all, diagnosis_categories[1], diagnosis_categories[0])) 


# In[5]:


def visualize_data(df):
    #type of classes
    print("classes are",df['diagnosis'].unique())
    classes = Counter(df['diagnosis'])
    plt.figure(figsize=(6,4))
    plt.bar('M',classes['M'],label='Malignant',color='r')
    plt.bar('B',classes['B'],label='Benign',color='b')
    plt.xlabel('diagnosis')
    plt.ylabel('count')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(),'plots','class_stats.png'), dpi=300)    


# In[6]:


def mean_parameter(df):
    #correlation matrix using heat map for features_mean
    features_mean= list(df.columns[1:11])
    corr = df[features_mean].corr() # .corr is used for find corelation
    fig, ax =plt.subplots(figsize=(14,14))
    sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm') 
    fig.savefig(os.path.join(os.getcwd(),'plots','correlation_matrix_mean.png'), dpi=300)

#observations:

   # 1)the radius, parameter and area are highly correlated as expected from their relation so from these we will use
   # anyone of them
   # 2)compactness_mean, concavity_mean and concavepoint_mean are highly correlated so we will use compactness_mean
   # from here
   # 3)so selected Parameter for use is perimeter_mean, texture_mean, compactness_mean, symmetry_mean


# In[7]:


def split_model(df):
    #now split our data into train and test
    train, test = train_test_split(df,test_size=0.2,shuffle=False)
    # Their dimensions are
    print("train data shape",train.shape)
    print("test data shape",test.shape)
    return train,test


# In[8]:


selected_parameter=[]


# In[9]:


def ttdata(train,test,selected_parameter):
    train_X = train[selected_parameter]# taking the training data input 
    train_y=train.diagnosis# This is output of our training data
    # same we have to do for test
    test_X= test[selected_parameter] # taking test data inputs
    test_y =test.diagnosis   #output value of test data
    return train_X,train_y,test_X,test_y


# In[10]:


def se_parameter(df):
    #correlation matrix using heat map for features_mean
    features_se= list(df.columns[11:21])
    corr = df[features_se].corr() # .corr is used for find corelation
    fig, ax =plt.subplots(figsize=(14,14))
    sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
    xticklabels= features_se, yticklabels= features_se,cmap= 'coolwarm') 
    fig.savefig(os.path.join(os.getcwd(),'plots','correlation_matrix_se.png'), dpi=300)


#observations:

   # 1)the radius, parameter and area are highly correlated as expected from their relation so from these we will use
   # anyone of them
   # 2)compactness_se, concavity_se and concavepoint_se are highly correlated so we will use compactness_se
   # from here
   # 3)so selected Parameter for use is perimeter_se, texture_se, compactness_se, symmetry_se


# In[11]:


def worst_parameter(df):
    #correlation matrix using heat map for features_mean
    features_worst= list(df.columns[21:31])
    corr = df[features_worst].corr() # .corr is used for find corelation
    fig, ax = plt.subplots(figsize=(14,14))
    sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
    xticklabels= features_worst, yticklabels= features_worst,cmap= 'coolwarm') 
    fig.savefig(os.path.join(os.getcwd(),'plots','correlation_matrix_worst.png'), dpi=300)



#observations:

   # 1)the radius, parameter and area are highly correlated as expected from their relation so from these we will use
   # anyone of them
   # 2)compactness_worst, concavity_worst and concavepoint_worst are highly correlated so we will use compactness_worst
   # from here
   # 3)so selected Parameter for use is perimeter_worst, texture_worst, compactness_worst, symmetry_worst

