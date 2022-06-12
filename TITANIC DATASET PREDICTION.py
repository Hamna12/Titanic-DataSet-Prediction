#!/usr/bin/env python
# coding: utf-8

# # TITANIC DATASET PREDICTION

# # PRESENTED BY: HAMNA QASEEM

# # IMPORTING ALL NECESSARY LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # DATA COLLECTION AND PROCESSING

# In[2]:


#load data from csv file to Pandas DataFrame
titanic_data=pd.read_csv('train.csv')


# In[3]:


#printing the first five rows of dataframe
titanic_data.head()


# In[4]:


#number of rows and columns
titanic_data.shape


# In[5]:


#getting some information about the data
titanic_data.info()


# In[6]:


#check the number of missing values in each column
titanic_data.isnull().sum()


# # Handling The Missing Values

# In[7]:


#drop cabin column from dataset
titanic_data=titanic_data.drop(columns='Cabin', axis=1)


# In[8]:


# replacing the missing values in 'Age' column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# In[9]:


# finding the mode value of 'Embarked' column
print(titanic_data['Embarked'].mode())


# In[10]:


print(titanic_data['Embarked'].mode()[0])


# In[11]:


# replace missing values in 'Embarked' column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)


# In[12]:


#check the number of missing values in each column
titanic_data.isnull().sum()


# # DATA ANALYSIS

# In[13]:


# Getting some statistical measures about data
titanic_data.describe()


# In[14]:


# finding the number of people survived or not survived
titanic_data['Survived'].value_counts()


# # DATA VISUALIZATION

# In[15]:


sns.set()


# In[16]:


# making a count plot for 'Survived' column
sns.countplot('Survived', data=titanic_data)


# In[17]:


# making a count plot for 'Sex' column
titanic_data['Sex'].value_counts()


# In[18]:


# making a count plot for 'Sex' column
sns.countplot('Sex', data=titanic_data)


# In[19]:


# number of survivors gender vise
sns.countplot('Sex', hue='Survived', data=titanic_data)


# In[20]:


# making a count plot for 'Pclass' column
sns.countplot('Pclass', data=titanic_data)


# In[21]:


sns.countplot('Pclass', hue='Survived', data=titanic_data)


# In[ ]:


# SUMMARY OF THIS WHOLE DATASET IS:
We have Training And Test Data and we Train the model and after that we Test it and make prediction on Target feature that is "Survived".
We Predict that how many passengers in Titanic was Survived and what was the distribution of Genders. More Passengers were Females or Males.


# # ENCODING THE CATEGORICAL COLUMN

# In[22]:


titanic_data['Sex'].value_counts()


# In[23]:


titanic_data['Embarked'].value_counts()


# In[24]:


# converting categorical columns
titanic_data.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)


# In[25]:


titanic_data.head()


# # SEPARATING FEATURES AND TARGET

# In[26]:


X=titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket','Survived'], axis=1)
y=titanic_data['Survived']


# In[27]:


print(X)


# In[28]:


print(y)


# # SPLITTING DATA INTO TRAINING DATA AND TEST DATA

# In[29]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=2)


# In[30]:


print(X.shape, X_test.shape, X_train.shape)


# # MODEL TRAINING

# # LOGISTIC REGRESSION MODEL

# In[31]:


model=LogisticRegression()


# In[32]:


#Training logistic regression model with training data
model.fit(X_train, y_train)


# # MODEL EVALUATION

# # ACCURACY SCORE

# In[33]:


# Accuracy on training data
X_train_prediction=model.predict(X_train)


# In[34]:


print(X_train_prediction)


# In[35]:


training_data_accuracy=accuracy_score(y_train, X_train_prediction)
print('Accuracy score of training data :', training_data_accuracy)


# In[36]:


# Accuracy on test data
X_test_prediction=model.predict(X_test)
print(X_test_prediction)


# In[37]:


test_data_accuracy=accuracy_score(y_test, X_test_prediction)
print('Accuracy score of test data :', test_data_accuracy)


# In[39]:


titanic_data.Survived.hist(bins=30, alpha=0.5)
plt.show()


# # THE END!
#           THANKyOU

# # TITANIC DATA SET:

# ![download%20%284%29.jpg](attachment:download%20%284%29.jpg)

# # OVERVIEW OF ENTIRE DATASET:

# ![image_2022-06-12_122523686.png](attachment:image_2022-06-12_122523686.png)

# In[40]:


# making a count plot for 'Sex' column
titanic_data['Sex'].value_counts()


# ###  After analyzing, from this graph we evaluate that huge amount of Males who were affected in Titanic Disaster. From 891 people 577 were Males who were died or not survived. And remaining were Females count of 314 who were Survived.

# # EVALUATION THROUGH CLASS WISE:

# ![download%20%284%29.png](attachment:download%20%284%29.png) 

# ![download%20%283%29.png](attachment:download%20%283%29.png)

# ### We can Evaluate that most of the Passengers were from 3rd class. And most passengers were affected in Disasters frpm Pclass 3.
# ### And the amount of male gender were affected most as compared to Females. 
# #### In Pclass 1 most of the passengers were Survived
# #### In Pclass 2 the ratio between Survived and Not Survived Passenger is too short.The amount of Survived Passengers is bit low.
# #### In Pclass 3 the ratio between Survived and Not Survived Passenger is too high. Most passengers were affected in this class. 

# # CONCLUSION:

# ### So we can conclude that the Death rate of Male Gender is more as compared to females in Titanic Disaster 

# # THE END!!
#    THANKyOU....
