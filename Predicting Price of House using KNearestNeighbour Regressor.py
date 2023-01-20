#!/usr/bin/env python
# coding: utf-8

# # Predicting Price of House using KNearestNeighbour Regressor

# ### Import Libraries

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


# ### load dataset

# In[2]:


url ="https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt"


# 1 . Use pandas to get some insights into the data.

# In[3]:


df = pd.read_csv(url)
df


# # 1.Use pandas to get some insights into the data.

# In[4]:


df = df.drop('Unnamed: 0', axis=1)


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df.isnull()


# In[9]:


df.dropna()


# In[10]:


df1=df.drop_duplicates()
df1


# In[11]:


df1.describe(include='all')


# # 2. Show some interesting visualization of the data.

# In[12]:


sns.pairplot(df1)


# Making a different - 2 group of column for check price.

# In[13]:


df_gp1 = df1.groupby(['Floor','Bedroom'], as_index = False)['Price'].mean()
plt.figure(figsize = (25,7))
plt.bar(df_gp1['Floor'].astype(str)+'_'+df_gp1['Bedroom'].astype(str),df_gp1['Price'],color='blue')
plt.xticks(rotation=90)
plt.title("(Floor and Bedroom) vs Price")
plt.xlabel('Floor and  Bedroom')
plt.ylabel('Price')
plt.legend(['Floor and Bedroom'])


# In[14]:


df_gp2 = df1.groupby(['TotalFloor','Bathroom'], as_index = False)['Price'].mean()
plt.figure(figsize = (25,7))
plt.bar(df_gp2['TotalFloor'].astype(str)+'_'+df_gp2['Bathroom'].astype(str),df_gp2['Price'],color='yellow')
plt.xticks(rotation=90)
plt.title("(TotalFloor and Bathroom) vs Price")
plt.xlabel('TotalFloor and  Bathroom')
plt.ylabel('Price')
plt.legend(['TotalFloor and Bathroom'])


# In[15]:


df_gp3 = df1.groupby(['Bathroom','Living.Room'], as_index = False)['Price'].mean()
plt.figure(figsize = (25,7))
plt.bar(df_gp3['Bathroom'].astype(str)+'_'+df_gp3['Living.Room'].astype(str),df_gp3['Price'],color='green')
plt.xticks(rotation=90)
plt.title("(Bathroom and Living.Room) vs Price")
plt.xlabel('Bathroom and  Living.Room')
plt.ylabel('Price')
plt.legend(['Bathroom and Living.Room'])


# # 3. Manage data for training & testing

# ## Split dataset

# In[16]:


X = df1.drop('Price', axis=1)
y = df1['Price']


# In[17]:


#from sklearn import preprocessing
#X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# In[19]:


print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[20]:


X


# In[21]:


y


# In[22]:


y.info()


# ### import the model

# In[23]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[24]:


#Train Model and Predict
k = 1  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
Pred_y = neigh.predict(X_test)
print("Accuracy of model at K = 1 is",metrics.accuracy_score(y_test, Pred_y))


# # 4 . Finding a better value of k.

# In[25]:


error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))


# In[26]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='black', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))


# In[27]:


acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='black', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

