#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


#Objective 1.
#Use this cell to import the Numpy (as np), Pandas (as pd), and YFinance (as yf) packages.

import numpy as np
import pandas as pd
import yfinance as yf 
import pandas_datareader as pdr
#Our work will also require some components of the Sklearn and Pandas_Datareader packages as imported below:

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from pandas_datareader import data as pdr
yf.pdr_override()


# In[15]:


#Objective 2.
#Select a stock symbol for a stock whose historical data is available on the Yahoo! Finance website.  Store the
#string of your chosen symbol to the new variable "stock_symbol".

stock_symbol = "UBER"


# In[16]:


#Objective 3.
#Use the function "pdr.get_data_yahoo(stock_symbol, start_date, end_date)" to generate a Pandas dataframe of
#historical stock data for your chosen stock.  Retain only the first four columns of the dataframe.  Be sure to 
#drop any rows containing NaN's, and take a peek at the resulting dataframe to make sure everything looks good.

df = pdr.get_data_yahoo(stock_symbol, "2016-1-1", "2021-1-1")
df = df.dropna()
df = df[["Open", "High", "Low", "Close"]]
df


# In[17]:


#Objective 4.
#Use the historical data to define some predictor variables.  Add these variables to the dataframe.  Include, at a
#minimum, (Predictor.I) the rolling average closing price over the past fifteen (15) days and (Predictor.II) the
#change in opening price over the past one (1) day.  Be sure to again drop any rows containing NaN's, and take a
#peek at the resulting dataframe to make sure everything looks good.  Store the dataframe as the new variable "X".

df['Close_15_Rolling'] = df["Close"].rolling(window=15).mean()
df['Open_1_Change'] = df["Open"].diff()
df = df.dropna()
X = df[["Close_15_Rolling", "Open_1_Change"]]
X


# In[18]:


#Objective 5.
#Define the target or dependent variable to be one (1) if the change in closing price over the past one (1) day is
#nonnegative and negative one (-1) if the change in closing price over the past one (1) day is negative.  This
#variable's values should be forward-looking (i.e., you should subtract today's price from tomorrow's price rather
#than subtracting yesterday's price from today's price).  Store the resulting values as the new variable "y".  You
#may find the "np.where(*args)" function to be useful.  Look it up in Numpy documentation for support.

y = np.where(-df["Close"].diff(-1) > 0, 1, -1)
y


# In[19]:


#Objective 6.
#Split the data into training and test sets, putting the first seventy percent (70%) of the data in the training
#set.

index = int(0.7*len(X))
X_train, X_test, y_train, y_test = X[:index], X[index:], y[:index], y[index:]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[20]:


#Objective 7.
#Instantiate the Logistic Regression model object, and use its ".fit(*args)" method to fit the model to the 
#training data.

logistic = LogisticRegression()
logistic = logistic.fit(X_train, y_train)


# In[21]:


#Objective 8.
#Examine the model's coefficients by using its ".coef_" method.

pd.DataFrame(zip(X.columns, np.transpose(logistic.coef_))), logistic.intercept_


# In[22]:


#Objective 9.
#Use the model's ".predict_proba(*args)" and ".predict(*args)" methods to generate predictions over the test set.

probabilities = logistic.predict_proba(X_test)
predictions = logistic.predict(X_test)


# In[23]:


#Objective 10.
#Use the function "metrics.confusion_matrix(*args)" to create a confusion matrix comparing the predicted and true
#classification labels over the test set.

metrics.confusion_matrix(y_test, predictions)


# In[24]:


#Objective 11.
#Calculate the model's accuracy on the test set using its ".score(*args)" method.

logistic.score(X_test, y_test)


# In[25]:


#Objective 12.
#Use five-fold cross validation to cross-check the accuracy of the model over different held-out test sets.  This
#is where you should use the function "cross_val_score(*args)".

cross_val = cross_val_score(logistic ,X_test, y_test)


# In[26]:


cross_val.mean()


# In[ ]:





# In[ ]:




