#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importing required libraries
import numpy as np
import pandas as pd, datetime
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from time import time
import os
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import  ARIMA
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from pandas import DataFrame
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# In[6]:


# Import datast 
store = pd.read_csv("C:/Users/ASUS/Downloads/store.csv")
train = pd.read_csv("C:/Users/ASUS/Downloads/train.csv/train.csv")
test = pd.read_csv("C:/Users/ASUS/Downloads/test.csv")
train.shape, test.shape, store.shape


# In[7]:


train.head()


# In[8]:


test.head()


# In[9]:


store.head()


# In[12]:


#1. Explamatory Data Analysis(EDA)


# In[13]:


train.shape


# In[18]:


# Extract Year, Month, Day, Wee columns 

train['SalesPerCustomer'] = train['Sales']/train['Customers']


# In[16]:


train.head()


# In[19]:


# Checking the data when the store is closed 
train_store_closed = train[(train.Open == 0)]
train_store_closed.head()


# In[20]:


# Check when the store was closed 
train_store_closed.hist('DayOfWeek')


# In[21]:


# Check whether there school was closed for holyday 
train_store_closed['SchoolHoliday'].value_counts().plot(kind='bar')


# In[22]:


# Check whether there school was closed for holyday 
train_store_closed['StateHoliday'].value_counts().plot(kind='bar')


# In[23]:


# Check the null values
# In here there is no null value 
train.isnull().sum()


# In[24]:


# Number of days with closed stores
train[(train.Open == 0)].shape[0]


# In[25]:


# Okay now check No. of dayes store open but sales zero ( It might be caused by external refurbishmnent)
train[(train.Open == 1) & (train.Sales == 0)].shape[0]


# In[26]:


# Work with store data 
store.head()


# In[27]:


# Check null values 
# Most of the columns has null values 

store.isnull().sum()


# In[28]:


# Replacing missing values for Competiton distance with median
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)


# In[29]:


# No info about other columns - so replcae by 0
store.fillna(0, inplace=True)


# In[30]:


# Again check it and now its okay 

store.isnull().sum().sum()


# In[31]:


# Work with test data 
test.head()


# In[32]:


# check null values ( Only one feature Open is empty)
test.isnull().sum()


# In[33]:


# Assuming stores open in test
test.fillna(1, inplace=True)


# In[34]:


# Again check 
test.isnull().sum().sum()


# In[35]:


# Join train and store table 
train_store_joined = pd.merge(train, store, on='Store', how='inner')
train_store_joined.head()


# In[36]:


train_store_joined.groupby('StoreType')['Customers', 'Sales', 'SalesPerCustomer'].sum().sort_values('Sales', ascending='desc')


# In[37]:


# Closed and zero-sales observations 
train_store_joined[(train_store_joined.Open == 0) | (train_store_joined.Sales==0)].shape


# In[38]:


# Open & Sales >0 stores
train_store_joined_open = train_store_joined[~((train_store_joined.Open ==0) | (train_store_joined.Sales==0))]
train_store_joined_open


# In[39]:


#Correlation Analysis


# In[40]:


plt.figure(figsize=(20, 10))
sns.heatmap(train_store_joined.corr(), annot=True)


# In[42]:


# Sales and trend over days
sns.factorplot(data= train_store_joined_open, x='DayOfWeek', y="Sales",
              hue='Promo')


# In[43]:


pd.plotting.register_matplotlib_converters()


# In[44]:


# Data Preparation: input should be float type 

# our Sales data is int type so lets make it float
train['Sales'] = train['Sales'] * 1.00

train['Sales'].head()


# In[45]:


train.Store.unique()


# In[50]:


# lets create a functions to test the stationarity 
def test_stationarity(timeseries):
    # Determine rolling statestics 
    roll_mean = timeseries.rolling(window=7).mean()
    roll_std = timeseries.rolling(window=7).std()
    
    # plotting rolling statestics 
    plt.subplots(figsize = (16, 6))
    orginal = plt.plot(timeseries.resample('w').mean(), color='blue',linewidth= 3, label='Orginal')
    roll_mean = plt.plot(roll_mean.resample('w').mean(), color='red',linewidth= 3, label='Rolling Mean')
    roll_mean = plt.plot(roll_std.resample('w').mean(), color='green',linewidth= 3, label='Rolling Std')
    
    plt.legend(loc='best')
    plt.show()
    
    # Performing Dickey-Fuller test 
    print('Result of Dickey-Fuller test:')
    result= adfuller(timeseries, autolag='AIC')
    
    print('ADF Statestics: %f' %result[0])
    print('P-value: %f' %result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(key, value)


# In[48]:


# lets create a functions to test the stationarity 
def test_stationarity(timeseries):
    # Determine rolling statestics 
    roll_mean = timeseries.rolling(window=7).mean()
    roll_std = timeseries.rolling(window=7).std()
    
    # plotting rolling statestics 
    plt.subplots(figsize = (16, 6))
    orginal = plt.plot(timeseries.resample('w').mean(), color='blue',linewidth= 3, label='Orginal')
    roll_mean = plt.plot(roll_mean.resample('w').mean(), color='red',linewidth= 3, label='Rolling Mean')
    roll_mean = plt.plot(roll_std.resample('w').mean(), color='green',linewidth= 3, label='Rolling Std')
    
    plt.legend(loc='best')
    plt.show()
    
    # Performing Dickey-Fuller test 
    print('Result of Dickey-Fuller test:')
    result= adfuller(timeseries, autolag='AIC')
    
    print('ADF Statestics: %f' %result[0])
    print('P-value: %f' %result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(key, value)


# In[51]:


# plotting trends and seasonality 

def plot_timeseries(sales,StoreType):

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    fig.set_figheight(6)
    fig.set_figwidth(20)

    decomposition= seasonal_decompose(sales, model = 'additive',freq=365)

    estimated_trend = decomposition.trend
    estimated_seasonal = decomposition.seasonal
    estimated_residual = decomposition.resid
    
    axes[1].plot(estimated_seasonal, 'g', label='Seasonality')
    axes[1].legend(loc='upper left');
    
    axes[0].plot(estimated_trend, label='Trend')
    axes[0].legend(loc='upper left');

    plt.title('Decomposition Plots')


# In[53]:


# Define the p, d and q parameters to take any value between 0 and 3
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA: ')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[ ]:





# In[ ]:





# In[ ]:




