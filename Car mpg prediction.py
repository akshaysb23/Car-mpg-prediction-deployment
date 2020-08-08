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

# In[4]:


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

# In[5]:


df=pd.read_csv('datasets_1489_2704_auto-mpg.csv')
df.head()


# Replacing the values of origin column to get better results
# 
# 1 to be replaced with American, 2 with European and 3 with Japan

# In[6]:


df['origin'].replace({1:'American',2:'European',3:'Japanese'},inplace=True)


#  # **Exploratory Data Analysis**

# In[7]:


df.info()


# In, the above cell, even though horsepower is numerical feature, it is shown as object

# In[8]:


df.describe()


# In[9]:


df['horsepower']=pd.to_numeric(df['horsepower'],errors='coerce')


# In[10]:


df['mpg'].plot(kind='kde')


# Distribution is almost normally distributed

# In[11]:


df['cylinders'].plot(kind='kde')


# Multiple peaks indicates, cylinders is discrete feature

# In[12]:


df['displacement'].plot(kind='kde')


# Most of cars have low to average displacements, few have very high displacements

# In[13]:


df['horsepower'].plot(kind='kde')


# Most of cars low to average horse powers. Few cars have high horse power

# In[14]:


df['weight'].plot(kind='kde')


# Most of  cars are heavy, there are very few cars which are light.

# In[15]:


df['acceleration'].plot(kind='kde')


# acceleration values are almostnormally distributed

# In[16]:


plt.figure(figsize=(8,8))
ax=sns.countplot(df['origin'])
for i in ax.patches:
    ax.annotate('{}'.format(i.get_height()),(i.get_x()+0.3,i.get_height()))


# There are 249 cars of american origin, 79 of Japenese origin and 70 cars of European origin

# In[17]:


plt.figure(figsize=(8,8))
ax=sns.barplot(x=df['origin'],y=df['weight'].median())


# Weight of cars of all origins are almost same

# In[18]:


acc=(df.groupby('origin')['acceleration'].median())
print(acc)
acc.plot(kind='bar')
plt.ylabel('Avg Acceleration')


# Japanese origin cars have highest acceleration whereas American origin cars have least acceleration among 3 origin

# In[19]:


hp=(df.groupby('origin')['horsepower'].median())
print(hp)
hp.plot(kind='bar')
plt.ylabel('Avg. HP')


# 1. Cars of American origin have highest horsepower. European and Japanese origin are no where near American cars in terms of horse power.
# 2. Even though Japanese origin cars have highest acceleration, they have least average horsepower
# 3. American origin cars have highest horse power. They outperform Japanese origin but they have lesser acceleration  

# In[20]:


mpg=(df.groupby('origin')['mpg'].median())
print(mpg)
mpg.plot(kind='bar')
plt.ylabel('Avg Mpg')


# American origin cars have lowet mpg among 3 origin cars. Japanese cars have highest mpg.

# In[21]:


plt.figure(figsize=(8,8))
ax=sns.countplot(df['model year'])
for i in ax.patches:
    ax.annotate('{}'.format(i.get_height()),(i.get_x()+0.3,i.get_height()))


# Most of the cars were introduced in the year 1973

# In[22]:


sns.scatterplot(x=df['weight'],y=df['mpg'])


# It can be seen that as weight increases, it requires higher amount of fuel to move as a result mpg reduces

# In[23]:


sns.scatterplot(x=df['weight'],y=df['horsepower'])


# If we need higher horsepower, we need more number of cylinders and increase in number of increases weight of car

# In[24]:


sns.scatterplot(x=df['horsepower'],y=df['mpg'])


# Increase in horsepower increases weight which reduces mpg

# In[25]:


sns.scatterplot(x=df['acceleration'],y=df['horsepower'])


# Higher the horsepower, more no. of cylinders, hence more displacement. Higher the displacement, lesser is the time to accelerate

# In[26]:


cor_mat=df.corr()
sns.heatmap(cor_mat,annot=True)


# 1. We can observe that there is good positive correlation between displacement and number of cylinders. As number of cylinders increases displacement increases.
# 2. Also there is good positive correlation between weight and number of cylinders, since increase in no. of cylinders increases dead weight of car which also increases displacement of car. Thus we can say that weight and displacement are directly correlated which is evident from above heatmap

# In[27]:


sns.pairplot(df,vars=['mpg','cylinders','displacement','horsepower','weight','acceleration'])


# # **Missing Value Imputation**
# Before using KNN imputer, let us create dummy columns for origin, model year and we will drop 'car name'

# In[28]:


col=['origin','model year']
df=pd.get_dummies(data=df,drop_first=True,columns=col)


# In[29]:


df.head()


# In[30]:


df.drop('car name',axis=1,inplace=True)


# In[31]:


imp=KNNImputer(missing_values=np.nan,n_neighbors=4)
df1=imp.fit_transform(df)


# In[32]:


df=pd.DataFrame(df1,columns=df.columns)


# In[33]:


df['horsepower'].unique()


# # **Model Building**

# In[34]:


# Base Model
x=df.drop('mpg',axis=1)
y=df['mpg']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=123)


# In[35]:


x_const=sm.add_constant(x_train)
model=sm.OLS(y_train,x_const).fit()
model.summary()


# In[36]:


vif = [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=x_train.columns).T


# In[37]:


x1=x.drop('horsepower',axis=1)


# In[38]:


x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.30,random_state=123)


# In[39]:


x_const=sm.add_constant(x_train)
model=sm.OLS(y_train,x_const).fit()
model.summary()


# In[40]:


vif = [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=x_train.columns).T


# In[41]:


x1=x1.drop('cylinders',axis=1)


# In[42]:


x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.30,random_state=123)


# In[43]:


x_const=sm.add_constant(x_train)
model=sm.OLS(y_train,x_const).fit()
model.summary()


# In[44]:


vif = [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=x_train.columns).T


# In[45]:


x1=x1.drop('displacement',axis=1)


# In[46]:


x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.30,random_state=123)


# In[47]:


x_const=sm.add_constant(x_train)
model=sm.OLS(y_train,x_const).fit()
model.summary()


# In[48]:


vif = [variance_inflation_factor(x_const.values, i) for i in range(x_const.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=x_train.columns).T


# In[49]:


lr=LinearRegression()
model=lr.fit(x_train,y_train)


# In[50]:


print(f'R^2 score for train: {lr.score(x_train, y_train)}')
print(f'R^2 score for test: {lr.score(x_test, y_test)}')


# In[51]:


y_pred=lr.predict(x_test)


# In[52]:


cv_results = cross_val_score(lr, x_train, y_train,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))


# In[53]:


mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)


# # **Boosting Regressors**

# In[54]:


GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))


# In[55]:


x_axis=np.arange(len(GB_bias))
plt.plot(x_axis,GB_bias)


# In[56]:


np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]


# In[57]:


np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]


# In[58]:


ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))


# In[59]:


x_axis=np.arange(len(ABLR_bias))
plt.plot(x_axis,ABLR_bias)


# In[60]:


np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]


# In[61]:


np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]


# # **Bagging Regressors**

# In[62]:


Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))


# In[63]:


np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]


# In[64]:


np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]


# In[65]:


RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))


# In[66]:


np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]


# In[67]:


np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]


# # **Standardising the data**

# In[68]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1234)
ss=StandardScaler()
x_s=ss.fit_transform(x)
x_trains=ss.fit_transform(x_train)
x_tests=ss.transform(x_test)


# In[69]:


lr=LinearRegression()
model=lr.fit(x_trains,y_train)


# In[70]:


print(f'R^2 score for train: {lr.score(x_trains, y_train)}')
print(f'R^2 score for test: {lr.score(x_tests, y_test)}')


# In[71]:


cv_results = cross_val_score(lr, x_trains, y_train,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))


# In[72]:


y_pred=lr.predict(x_tests)


# In[73]:


mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)


# # **Boosting Regressor**

# In[74]:


GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,x_trains,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))


# In[75]:


np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]


# In[76]:


np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]


# In[77]:


ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,x_trains,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))


# In[78]:


np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]


# In[79]:


np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]


# # **Bagging Regressor**

# In[80]:


Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,x_train,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))


# In[81]:


np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]


# In[82]:


np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]


# In[83]:


RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,x_trains,y_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))


# In[84]:


np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]


# In[85]:


np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]


# # **Feature Elimination**

# In[86]:


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


# In[87]:


X_new=x[selected_features]
X_new.head()


# In[88]:


x_train1,x_test1,y_train1,y_test1=train_test_split(X_new,y,test_size=0.30,random_state=1234)


# In[89]:


lr=LinearRegression()
model=lr.fit(x_train1,y_train1)


# In[90]:


print(f'R^2 score for train: {lr.score(x_train1, y_train1)}')
print(f'R^2 score for test: {lr.score(x_test1, y_test1)}')


# In[91]:


cv_results = cross_val_score(lr, x_train1, y_train1,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))


# In[92]:


y_pred=lr.predict(x_test1)


# In[93]:


mse=mean_squared_error(y_test1,y_pred)
rmse=np.sqrt(mse)
print(rmse)


# # **Boosting Regressor**

# In[94]:


GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,x_train1,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))


# In[95]:


np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]


# In[96]:


np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]


# In[97]:


ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,x_train1,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))


# In[98]:


np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]


# In[99]:


np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]


# # **Bagging Regressor**

# In[100]:


Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,x_train1,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))


# In[101]:


np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]


# In[102]:


np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]


# In[103]:


RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,x_train1,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))


# In[104]:


np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]


# In[105]:


np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]


# # **Standardising the selected features**

# In[106]:


x_news=ss.fit_transform(X_new)
x_train1s=ss.fit_transform(x_train1)
x_test1s=ss.transform(x_test1)


# In[107]:


lr=LinearRegression()
model=lr.fit(x_train1s,y_train1)


# In[108]:


print(f'R^2 score for train: {lr.score(x_train1s, y_train1)}')
print(f'R^2 score for test: {lr.score(x_test1s, y_test1)}')


# In[109]:


cv_results = cross_val_score(lr, x_train1s, y_train1,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))


# In[110]:


y_pred=lr.predict(x_test1s)


# In[111]:


mse=mean_squared_error(y_test1,y_pred)
rmse=np.sqrt(mse)
print(rmse)


# # **Boosting Regressor**

# In[112]:


GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,x_train1s,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))


# In[113]:


np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]


# In[114]:


np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]


# In[115]:


ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,x_train1s,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))


# In[116]:


np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]


# In[117]:


np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]


# # **Bagging Regressor**

# In[118]:


Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,x_train1s,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))


# In[119]:


np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]


# In[120]:


np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]


# In[121]:


RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,x_train1s,y_train1,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))


# In[122]:


np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]


# In[123]:


np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]


# # Ridge and Lasso Regression

# In[124]:


Rd=Ridge(alpha=0.5,normalize=True)
Ls=Lasso(alpha=0.1,normalize=True)
En=ElasticNet(alpha=0.01,l1_ratio=0.919,normalize=True)
models = []
models.append(('Ridge',Rd))
models.append(('Lasso',Ls))
models.append(('Elastic',En))


# In[125]:


results = []
names = []
for name, model in models:
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    cv_results = cross_val_score(model, x, y,cv=kfold, scoring='neg_mean_squared_error')
    results.append(np.sqrt(np.abs(cv_results)))
    names.append(name)
    print("%s: %f (%f)" % (name, np.mean(np.sqrt(np.abs(cv_results))),np.std(np.sqrt(np.abs(cv_results)),ddof=1)))
   # boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[126]:


results = []
names = []
for name, model in models:
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    cv_results = cross_val_score(model, X_new, y,cv=kfold, scoring='neg_mean_squared_error')
    results.append(np.sqrt(np.abs(cv_results)))
    names.append(name)
    print("%s: %f (%f)" % (name, np.mean(np.sqrt(np.abs(cv_results))),np.std(np.sqrt(np.abs(cv_results)),ddof=1)))
   # boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[127]:


df


# # **Polynomial Regression**

# In[128]:


x_final=df.drop('mpg',axis=1)


# In[129]:


x_qr=x_final[['displacement','horsepower','weight','acceleration']]


# In[130]:


qr=PolynomialFeatures(degree=2)
x_qr=qr.fit_transform(x_qr)


# In[131]:


x_qr_df=pd.DataFrame(x_qr)
x_qr_df.head()


# In[132]:


x_qr_df=x_qr_df.drop(0,axis=1)


# In[133]:


idx=np.arange(x_final.shape[0])


# In[134]:


y.index=idx


# In[135]:


x_final.index=idx


# In[136]:


x_qr_df=pd.concat([x_final,x_qr_df,y],axis=1)


# In[137]:


x_qr_df.head()


# In[138]:


x_qr_df.drop(['displacement','horsepower','weight','acceleration'],axis=1,inplace=True)


# In[139]:


x_qr_df.columns


# In[140]:


x_qr_df.columns=['cylinders', 'origin_European', 'origin_Japanese','model year_71',   'model year_72',   'model year_73',
                 'model year_74',   'model year_75',   'model year_76','model year_77',   'model year_78',   'model year_79',
                 'model year_80',   'model year_81',   'model year_82','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10',
                 'f11','f12','f13','f14','mpg']


# In[141]:


x_qr=x_qr_df.drop('mpg',axis=1)
y_qr=x_qr_df['mpg']


# In[142]:


qr=LinearRegression()
models = []
models.append(('Ridge',Rd))
models.append(('Lasso',Ls))
models.append(('Elastic',En))
models.append(('Quadratic',qr))


# In[143]:


results = []
names = []
for name, model in models:
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    cv_results = cross_val_score(model, x_qr, y_qr,cv=kfold, scoring='neg_mean_squared_error')
    results.append(np.sqrt(np.abs(cv_results)))
    names.append(name)
    print("%s: %f (%f)" % (name, np.mean(np.sqrt(np.abs(cv_results))),np.std(np.sqrt(np.abs(cv_results)),ddof=1)))
   # boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[144]:


xqr_train,xqr_test,yqr_train,yqr_test=train_test_split(x_qr,y_qr,test_size=0.30,random_state=1234)
cv_results = cross_val_score(qr, xqr_train, yqr_train,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))


# In[145]:


y_pred=lr.predict(xqr_test)
mse=mean_squared_error(yqr_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)


# In[146]:


GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,xqr_train,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))


# In[147]:


np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]


# In[148]:


np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]


# In[149]:


ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,xqr_train,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))


# In[150]:


np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]


# In[151]:


np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]


# # **Bagging Regressor**

# In[152]:


Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,xqr_train,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))


# In[153]:


np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]


# In[154]:


np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]


# In[155]:


RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,xqr_train,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))


# In[156]:


np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]


# In[157]:


np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]


# # **Standardising the polynomial features**

# In[158]:


xqr_s=ss.fit_transform(x_qr)
xqr_trains=ss.fit_transform(xqr_train)
xqr_tests=ss.transform(xqr_test)


# In[159]:


cv_results = cross_val_score(qr, xqr_trains, yqr_train,cv=5, scoring='neg_mean_squared_error')
print(np.mean(np.sqrt(np.abs(cv_results))))
print(np.std(np.sqrt(np.abs(cv_results)),ddof=1))


# In[160]:


y_pred=qr.predict(xqr_tests)
mse=mean_squared_error(yqr_test,y_pred)
rmse=np.sqrt(mse)
print(rmse)


# In[ ]:


GB_bias=[]
GB_var=[]
for n in np.arange(1,150):
    GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(GB,xqr_trains,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    GB_bias.append(np.mean(rmse))
    GB_var.append(np.std(rmse,ddof=1))


# In[ ]:


np.argmin(GB_bias),GB_bias[np.argmin(GB_bias)],GB_var[np.argmin(GB_bias)]


# In[ ]:


np.argmin(GB_var),GB_var[np.argmin(GB_var)],GB_bias[np.argmin(GB_var)]


# In[ ]:


ABLR_bias=[]
ABLR_var=[]
for n in np.arange(1,150):
    ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(ABLR,xqr_trains,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    ABLR_bias.append(np.mean(rmse))
    ABLR_var.append(np.std(rmse,ddof=1))


# In[ ]:


np.argmin(ABLR_bias), ABLR_bias[np.argmin(ABLR_bias)],ABLR_var[np.argmin(ABLR_bias)]


# In[ ]:


np.argmin(ABLR_var), ABLR_var[np.argmin(ABLR_var)],ABLR_bias[np.argmin(ABLR_var)]


# In[ ]:


Bag_bias=[]
Bag_var=[]
for n in np.arange(1,150):
    Bag=BaggingRegressor(base_estimator=lr,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,xqr_trains,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    Bag_bias.append(np.mean(rmse))
    Bag_var.append(np.std(rmse,ddof=1))


# In[ ]:


np.argmin(Bag_bias),Bag_bias[np.argmin(Bag_bias)],Bag_var[np.argmin(Bag_bias)]


# In[ ]:


np.argmin(Bag_var),Bag_var[np.argmin(Bag_var)],Bag_bias[np.argmin(Bag_var)]


# In[ ]:


RF_bias=[]
RF_var=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    scores=cross_val_score(RF,xqr_trains,yqr_train,cv=5,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    RF_bias.append(np.mean(rmse))
    RF_var.append(np.std(rmse,ddof=1))


# In[ ]:


np.argmin(RF_bias),RF_bias[np.argmin(RF_bias)],RF_var[np.argmin(RF_bias)]


# In[ ]:


np.argmin(RF_var),RF_var[np.argmin(RF_var)],RF_bias[np.argmin(RF_var)]


# # Below are the results of all combinations:
# 
# 1. Removing features which are multicollinear:
#     1. Linear Regression - rmse = 3.18 (0.25)
#     2. GB Regressor - n_estimator = 21, rmse = 3.76 (0.15)
#     3. AdaBoost Regressor - n_estimator = 2, rmse = 3.26 (0.16)
#     4. Bagging LR - n_estimator = 1, rmse = 3.25 (0.16)
#     5. Random Forest - n_estimator = 93, rmse = 3.52 (0.15)
#     
# 2. Removing features which are multicollinear and standardising the data:
#     1. Linear Regression - rmse = 3.19 (0.19)
#     2. GB Regressor - n_estimator = 9, rmse = 4.62 (0.22)
#     3. AdaBoost Regressor - n_estimator = 3, rmse = 3.16 (0.14)
#     4. Bagging LR - n_estimator = 135, rmse = 3.19 (0.18)
#     5. Random Forest - n_estimator = 8, rmse = 3.58 (0.42)
#     
# 3. Removing features with P-value greater than 0.05:
#     1. Linear Regression - rmse = 3.198 (0.17)
#     2. GB Regressor - n_estimator = 130, rmse = 3.195 (0.58)
#     3. AdaBoost Regressor - n_estimator = 3, rmse = 3.155 (0.14)
#     4. Bagging LR - n_estimator = 62, rmse = 3.199 (0.168)
#     5. Random Forest - n_estimator = 135, rmse = 3.39 (0.46)
#     
# 4. Removing Features with P-Value greater than 0.05 and standardising the data:
#     1. Linear Regression - rmse = 3.198 (0.17)
#     2. GB Regressor - n_estimator = 6, rmse = 5.176 (0.26)
#     3. AdaBoost Regressor - n_estimator = 3, rmse = 3.155 (0.14)
#     4. Bagging LR - n_estimator = 62, rmse = 3.199 (0.168)
#     5. Random Forest - n_estimator = 135, rmse = 3.395 (0.457)
#     
# 5. Polynomial Features
#     1. Linear Regression - rmse = 2.84 (0.34)
#     2. GB Regressor - n_estimator = 100, rmse = 3.32 (0.30)
#     3. AdaBoost Regressor - n_estimator = 1, rmse = 3.11 (0.28)
#     4. Bagging LR - n_estimator = 116, rmse = 2.83 (0.33)
#     5. Random Forest - n_estimator = 135, rmse = 3.395 (0.457)
#     
# **It can be seen that Ada Boost regressor with features less than 0.05 has lowest RMSE, hence this can be used for our predictions**

# In[166]:


ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=3,random_state=0)
ABLR.fit(X_new,y)


# In[ ]:





# In[167]:


pickle.dump(ABLR,open('trial.pkl','wb'))


# In[168]:


trial=pickle.load(open('trial.pkl','rb'))


# In[169]:


X_new.columns


# In[170]:


app=Flask(__name__)
trial=pickle.load(open('trial.pkl','rb'))


# In[ ]:




