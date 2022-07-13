#!/usr/bin/env python
# coding: utf-8

# In[47]:


#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[48]:


#Importing dataset
dataset = pd.read_csv(r"G:\Other computers\My Laptop\Gsolar_Rimsha\Rimsha_AdditionalPractice\Projects\Linear Regression project with Python\aids.csv")


# In[51]:


print(dataset.describe().round(decimals = 0))
print(dataset.head())
dataset.shape


# In[52]:


x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# In[54]:


## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
x_train.shape


# In[55]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[41]:


y_pred = regressor.predict(x_test)


# In[59]:


# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Death vs Years (Test set)')
plt.xlabel('Years')
plt.ylabel('Death')
plt.show()


# In[65]:


print(regressor.predict([[1985]]))


# In[ ]:




