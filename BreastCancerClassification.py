#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from collections import Counter

import mode
from mode import process_data,visualize_data,mean_parameter,se_parameter,worst_parameter
from mode import split_model,ttdata

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,auc,roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)


# # Loading the Data

# In[2]:


df= pd.read_csv('breast-cancer.csv')


# In[3]:


print("Column names in data:", df.columns)


# # Pre-processing and Visualizing data

# In[4]:


# Creating a folder if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')


# In[5]:


df.drop(["Unnamed: 32","id"],axis=1,inplace=True)


# In[6]:


process_data(df)


# In[7]:


visualize_data(df)


# In[8]:


#converting labels into integer values for binary classification
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})


# # For Mean Parameters

# In[9]:


mean_parameter(df)


# In[10]:


selected_parameter_mean = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']


# In[11]:


mean_acc=[]
mean_cvs=[]


# # Splitting the model for training and testing with a test_size of 20 %

# In[12]:


train,test=split_model(df)


# In[13]:


train_X,train_y,test_X,test_y=ttdata(train,test,selected_parameter_mean)


# # Logistic Regression Classifier

# In[14]:


np.random.seed(1)
clf = LogisticRegression()
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
mean_acc.append(accuracy_score(prediction,test_y))
mean_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Nearest Neighbors Classifier

# In[15]:


np.random.seed(1)
clf = KNeighborsClassifier(n_neighbors=15,p=2,algorithm='kd_tree',leaf_size=20)
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
mean_acc.append(accuracy_score(prediction,test_y))
mean_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Decision Tree Classifier

# In[16]:


np.random.seed(1)
clf = DecisionTreeClassifier(splitter='best')
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
mean_acc.append(accuracy_score(prediction,test_y))
mean_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Random Forest Classifier

# In[17]:


np.random.seed(1)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
mean_acc.append(accuracy_score(prediction,test_y))
mean_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Naive Bayes Classifier

# In[18]:


np.random.seed(1)
clf = GaussianNB(var_smoothing=1e-7)
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
mean_acc.append(accuracy_score(prediction,test_y))
mean_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # AdaBoost classifier

# In[19]:


np.random.seed(1)
clf=AdaBoostClassifier(n_estimators=20,learning_rate=1)
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
mean_acc.append(accuracy_score(prediction,test_y))
mean_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Radial Basis Function Classifier

# In[20]:


np.random.seed(1)
kernel = 1.0 * RBF([1.0])
clf = GaussianProcessClassifier(kernel=kernel,max_iter_predict=100)
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
mean_acc.append(accuracy_score(prediction,test_y))
mean_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Support Vector Machine Classifier

# In[21]:


np.random.seed(1)
clf = SVC(C=4.0,kernel='linear')
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
mean_acc.append(accuracy_score(prediction,test_y))
mean_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Quadratic Discriminant Analysis Classifier

# In[22]:


np.random.seed(1)
clf = QuadraticDiscriminantAnalysis(reg_param=1.0)
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
mean_acc.append(accuracy_score(prediction,test_y))
mean_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # MultiLayer Perceptron Classifier

# In[23]:


np.random.seed(1)
clf = MLPClassifier(hidden_layer_sizes=(32,),alpha=1, max_iter=1000,activation='relu',solver='adam'
                    ,shuffle=False)
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
mean_acc.append(accuracy_score(prediction,test_y))
mean_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # For Standard Error(SE) Parameters

# In[24]:


se_parameter(df)


# In[25]:


selected_parameter_se = ['texture_se','perimeter_se','smoothness_se','compactness_se','symmetry_se']


# In[26]:


se_acc=[]
se_cvs=[]


# # Splitting the model for training and testing with a test_size of 20 %

# In[27]:


train,test=split_model(df)


# In[28]:


train_X,train_y,test_X,test_y=ttdata(train,test,selected_parameter_se)


# # Logistic Regression Classifier

# In[29]:


np.random.seed(1)
clf = LogisticRegression(tol=1e-4)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
se_acc.append(accuracy_score(prediction,test_y))
se_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Nearest Neighbors Classifier

# In[30]:


np.random.seed(1)
clf = KNeighborsClassifier(n_neighbors=15,p=2,algorithm='ball_tree',leaf_size=20)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
se_acc.append(accuracy_score(prediction,test_y))
se_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Decision Tree Classifier

# In[31]:


np.random.seed(1)
clf = DecisionTreeClassifier(splitter='best')
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
se_acc.append(accuracy_score(prediction,test_y))
se_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Random Forest Classifier

# In[32]:


np.random.seed(1)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
se_acc.append(accuracy_score(prediction,test_y))
se_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Naive Bayes Classifier

# In[33]:


np.random.seed(1)
clf = GaussianNB(var_smoothing=1e-7)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
se_acc.append(accuracy_score(prediction,test_y))
se_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # AdaBoost classifier

# In[34]:


np.random.seed(1)
clf=AdaBoostClassifier(n_estimators=20,learning_rate=1)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
se_acc.append(accuracy_score(prediction,test_y))
se_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Radial Basis Function Classifier

# In[35]:


np.random.seed(1)
kernel = 1.0 * RBF([1.0])
clf = GaussianProcessClassifier(kernel=kernel,max_iter_predict=100)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
se_acc.append(accuracy_score(prediction,test_y))
se_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Support Vector Machine Classifier

# In[36]:


np.random.seed(1)
clf = SVC(C=4.0,kernel='linear')
clf.fit(train_X,train_y)
prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
se_acc.append(accuracy_score(prediction,test_y))
se_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Quadratic Discriminant Analysis Classifier

# In[37]:


np.random.seed(1)
clf = QuadraticDiscriminantAnalysis(reg_param=1.0)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
se_acc.append(accuracy_score(prediction,test_y))
se_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # MultiLayer Perceptron Classifier

# In[38]:


np.random.seed(1)
clf = MLPClassifier(hidden_layer_sizes=(32,),alpha=1, max_iter=1000,activation='relu',solver='adam',
                    shuffle=False)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
se_acc.append(accuracy_score(prediction,test_y))
se_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # For Worst Parameters

# In[39]:


worst_parameter(df)


# In[40]:


selected_parameter_worst = ['texture_worst','perimeter_worst','smoothness_worst','compactness_worst','symmetry_worst']


# In[41]:


worst_acc=[]
worst_cvs=[]


# # Splitting the model for training and testing with a test_size of 20 %

# In[42]:


train,test=split_model(df)


# In[43]:


train_X,train_y,test_X,test_y=ttdata(train,test,selected_parameter_worst)


# # Logistic Regression Classifier

# In[44]:


np.random.seed(1)
clf = LogisticRegression()
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
worst_acc.append(accuracy_score(prediction,test_y))
worst_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Nearest Neighbors Classifier

# In[45]:


np.random.seed(1)
clf = KNeighborsClassifier(n_neighbors=15,p=2,algorithm='kd_tree',leaf_size=20)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
worst_acc.append(accuracy_score(prediction,test_y))
worst_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Decision Tree Classifier

# In[46]:


np.random.seed(1)
clf = DecisionTreeClassifier(splitter='best')
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
worst_acc.append(accuracy_score(prediction,test_y))
worst_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Random Forest Classifier

# In[47]:


np.random.seed(1)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
worst_acc.append(accuracy_score(prediction,test_y))
worst_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Naive Bayes Classifier

# In[48]:


np.random.seed(1)
clf = GaussianNB(var_smoothing=1e-7)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
worst_acc.append(accuracy_score(prediction,test_y))
worst_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # AdaBoost classifier

# In[49]:


np.random.seed(1)
clf=AdaBoostClassifier(n_estimators=20,learning_rate=1)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
worst_acc.append(accuracy_score(prediction,test_y))
worst_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Radial Basis Function Classifier

# In[50]:


np.random.seed(1)
kernel = 1.0 * RBF([1.0])
clf = GaussianProcessClassifier(kernel=kernel,max_iter_predict=100)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
worst_acc.append(accuracy_score(prediction,test_y))
worst_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Support Vector Machine Classifier

# In[51]:


np.random.seed(1)
clf = SVC(C=3.0,kernel='linear')
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
worst_acc.append(accuracy_score(prediction,test_y))
worst_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Quadratic Discriminant Analysis Classifier

# In[52]:


np.random.seed(1)
clf = QuadraticDiscriminantAnalysis(reg_param=1.0)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
worst_acc.append(accuracy_score(prediction,test_y))
worst_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # MultiLayer Perceptron Classifier

# In[53]:


np.random.seed(1)
clf = MLPClassifier(hidden_layer_sizes=(32,),alpha=1, max_iter=1000,activation='relu',solver='adam',
                   shuffle=False)
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)
scores = cross_val_score(clf,train_X,train_y, cv=5)
worst_acc.append(accuracy_score(prediction,test_y))
worst_cvs.append(np.mean(scores))
print("model accuracy: {0:.2%}".format(accuracy_score(prediction, test_y)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
    #print("model accuracy:",metrics.accuracy_score(prediction)


# # Visualizing the performance matrix

# In[54]:


index=['Logistic Regression','Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
       'AdaBoost','Radial Basis Function','Support Vector Machine', 'Quadratic Discriminant Analysis',
       'MultiLayer Perceptron']
data={'mean_accuracy':mean_acc,'mean_cvs':mean_cvs,'se_accuracy':se_acc,'se_cvs':se_cvs,
      'worst_accuracy':worst_acc,'worst_cvs':worst_cvs}
dataframe = pd.DataFrame(data,index=index)
dataframe


# # As it can be seen that the accuracy is most in SVM Classifier, lets use it to predict the testing dataset.

# In[55]:


train,test=split_model(df)


# In[56]:


selected_parameter_worst = ['texture_worst','perimeter_worst','smoothness_worst','compactness_worst','symmetry_worst']


# In[57]:


train_X,train_y,test_X,test_y=ttdata(train,test,selected_parameter_worst)


# In[58]:


clf = SVC(C=3.0,kernel='linear')
clf.fit(train_X,train_y)

prediction=clf.predict(test_X)


# In[59]:


prediction


# # Calculation of loss function

# In[60]:


loss = 0

for predicted_value,actual_value in zip(prediction,test_y):
    if predicted_value == actual_value:
        pass
    else:
        loss += 1

print("Loss :",loss/len(prediction)*100,"%")


# # Confusion Matrix

# In[64]:


#create confusion matrix
cm = confusion_matrix(test_y, prediction)
# prepare heatmap
fig, ax =plt.subplots(figsize=(6,4))
sns.heatmap(cm, annot= True,cmap= 'coolwarm')
fig.savefig(os.path.join(os.getcwd(),'plots','confusion_matrix.png'), dpi=300)


# In[65]:


print("Accuracy score %f" % accuracy_score(test_y, prediction))
print(classification_report(test_y, prediction))


# In[66]:


fpr,tpr,_ = roc_curve(test_y,prediction)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(14,12))
plt.title('Receiver Operating Characteristic')
sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.savefig(os.path.join(os.getcwd(),'plots','roc.png'), dpi=300) 

