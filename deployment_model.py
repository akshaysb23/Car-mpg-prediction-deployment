#!/usr/bin/env python
# coding: utf-8

# **In this kernel, I have used 4 different cases with 5 algorithms to predict best possible RMSE value along with varience error**
# Following are the four different cases that are tried:
# 1. Removing features which are multicollinear
# 2. Standardising the data after removing features which are collinear
# 3. Removing features with P-value greater than 0.05
# 4. Removing features with P-value greater than 0.05 and standardising the data
# 
# Below are the five algorithms that are used in our anaysis:
# 
# 1. Linear Regression
# 2. Gradient Boost Regressor
# 3. AdaBoost Regressor
# 4. Linear Regressor Bagging
# 5. Random Forest regressor

# # Data Information
# 
# Data contains technical information about different cars from which we need to predict mpg of car.
# Following are the information about the features:
# 
# 1. mpg: Miles per gallon run by car
# 2. cylinders: No. of cylinders in engine
# 3. displacement: engine displacement in cubic inches
# 4. horsepower: Horse power of particular car
# 5. weight: Dead weight of car in lbs
# 6. acceleration: Time taken for car to reach from 0 mph to 60 mph
# 7. model year: Year in which car was released
# 8. origin: Country of origin 1 - American, 2 - European, 3 - Japenese
# 9. car name: Name of car

# # **Importing Necessary Libraries**

# In[181]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,VotingRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as smf
from flask import Flask, request, jsonify,render_template
import pickle


# # **Importing Data**

# In[182]:


df=pd.read_csv('datasets_1489_2704_auto-mpg.csv')
df.head()


# Replacing the values of origin column to get better results
# 
# 1 to be replaced with American, 2 with European and 3 with Japan

# In[183]:


df['origin'].replace({1:'American',2:'European',3:'Japanese'},inplace=True)


# # **Missing Value Imputation**
# Before using KNN imputer, let us create dummy columns for origin, model year and we will drop 'car name'

# In[184]:


col=['origin','model year']
df=pd.get_dummies(data=df,drop_first=True,columns=col)


# In[185]:


df.head()


# In[186]:


df.drop('car name',axis=1,inplace=True)


# In[187]:


df['horsepower'].replace({'?':np.nan},inplace=True)


# In[188]:


imp=KNNImputer(missing_values=np.nan,n_neighbors=4)
df1=imp.fit_transform(df)


# In[189]:


df=pd.DataFrame(df1,columns=df.columns)


# # **Feature Elimination**

# In[190]:


cols = list(x.columns)
pmax = 1
while (len(cols)>0):
    p= []
    x = x[cols]
    Xc = sm.add_constant(x)
    model = sm.OLS(y,Xc).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features = cols
print(selected_features)


# In[191]:


X_new=x[selected_features]
X_new.head()


# In[192]:


x_train1,x_test1,y_train1,y_test1=train_test_split(X_new,y,test_size=0.30,random_state=1234)


# In[194]:


lr=LinearRegression()
ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=3,random_state=0)
ABLR.fit(X_new,y)


# In[195]:


pickle.dump(ABLR,open('deployment.pkl','wb'))


# In[ ]:




