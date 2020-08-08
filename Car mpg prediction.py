# # **Importing Necessary Libraries**
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

df=pd.read_csv('datasets_1489_2704_auto-mpg.csv')
df.head()

df['origin'].replace({1:'American',2:'European',3:'Japanese'},inplace=True)




df.info()



df.describe()




df['horsepower']=pd.to_numeric(df['horsepower'],errors='coerce')





df['mpg'].plot(kind='kde')



df['cylinders'].plot(kind='kde')




df['displacement'].plot(kind='kde')



df['horsepower'].plot(kind='kde')




df['weight'].plot(kind='kde')




df['acceleration'].plot(kind='kde')



plt.figure(figsize=(8,8))
ax=sns.countplot(df['origin'])
for i in ax.patches:
    ax.annotate('{}'.format(i.get_height()),(i.get_x()+0.3,i.get_height()))



plt.figure(figsize=(8,8))
ax=sns.barplot(x=df['origin'],y=df['weight'].median())



acc=(df.groupby('origin')['acceleration'].median())
print(acc)
acc.plot(kind='bar')
plt.ylabel('Avg Acceleration')



hp=(df.groupby('origin')['horsepower'].median())
print(hp)
hp.plot(kind='bar')
plt.ylabel('Avg. HP')



mpg=(df.groupby('origin')['mpg'].median())
print(mpg)
mpg.plot(kind='bar')
plt.ylabel('Avg Mpg')

plt.figure(figsize=(8,8))
ax=sns.countplot(df['model year'])
for i in ax.patches:
    ax.annotate('{}'.format(i.get_height()),(i.get_x()+0.3,i.get_height()))


sns.scatterplot(x=df['weight'],y=df['mpg'])


sns.scatterplot(x=df['weight'],y=df['horsepower'])




sns.scatterplot(x=df['horsepower'],y=df['mpg'])




sns.scatterplot(x=df['acceleration'],y=df['horsepower'])




cor_mat=df.corr()
sns.heatmap(cor_mat,annot=True)




sns.pairplot(df,vars=['mpg','cylinders','displacement','horsepower','weight','acceleration'])



col=['origin','model year']
df=pd.get_dummies(data=df,drop_first=True,columns=col)


df.head()




df.drop('car name',axis=1,inplace=True)




imp=KNNImputer(missing_values=np.nan,n_neighbors=4)
df1=imp.fit_transform(df)



df=pd.DataFrame(df1,columns=df.columns)



df['horsepower'].unique()


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



X_new=x[selected_features]
X_new.head()



x_train1,x_test1,y_train1,y_test1=train_test_split(X_new,y,test_size=0.30,random_state=1234)


ABLR=AdaBoostRegressor(base_estimator=lr,n_estimators=3,random_state=0)
ABLR.fit(X_new,y)


