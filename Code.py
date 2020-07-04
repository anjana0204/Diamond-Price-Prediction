# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:55:04 2020

@author: Akshay
"""

"""Importing libraries"""
import numpy as np
import pandas as pd
import math 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 
import matplotlib.pyplot as plt
import seaborn as sns


"""Importing dataset"""
df = pd.read_csv("C:/Users/Akshay/Desktop/Anjana/Python Projects/Diamond Price Prediction/diamonds.csv")
diamonds = df.copy()

"""Observing the dataset"""
df.head()

"""
Categorical Features - Cut, Color, Clarity.

Numerical Features - Carat, Depth , Table , Price , X , Y, Z.

Target Variable - Price

"""

""" Dropping the unnecessary variable Unnamed:0 """

df=df.drop(['Unnamed: 0'] , axis=1)
df.head()
df.shape #53940 rows and 10 columns
df.dtypes

"""Checking for missing values"""
df.isnull().sum()  # Nomissing values found

df.describe() 
#Min value of length,width and height is 0.00, which is impossible

""" Dropping rows with either height, width or length equal to 0 """
df=df[(df[['x','y','z']]!=0).all(axis=1)]
df.describe()

"""Correlation between features"""
corr = df.corr()
sns.heatmap(data=corr, square=True , annot=True, cbar=True)

"""
Observations
1. Depth is inversely related to Price.
2. The Price of the Diamond is highly correlated to Carat,length, width and height.
3. The Weight (Carat) of a diamond has the most significant impact on its Price along with length, width and height.
4. The Length(x) , Width(y) and Height(z) seems to be higly correlated to each other.

"""

"""Visualization"""
#Carat
sns.kdeplot(df['carat'], shade=True , color='r')
sns.scatterplot(x='carat',y ='price',data=df )
#price increases exponentially with carat and there are only a few data points wuth high carat due to its rarity

#Cut
sns.factorplot(x='cut', data=df , kind='count')
sns.factorplot(x='cut', y='price',data=df , kind='box')
#ideal cut diamonds are the highest in number and premium cut diamons are most expensive

#Color
sns.factorplot(x='color', data=df , kind='count')
sns.factorplot(x='color', y='price',data=df , kind='box')
# G colored diamonds are highest in number and I and J colored diamonds are more expensive

#Clarity
sns.factorplot(x='clarity', data=df , kind='count')
sns.factorplot(x='clarity', y='price',data=df , kind='box')
#SI1 and VS2 are the clarity type with highest numbr of diamonds
#VS1 and VS2 are the clarity with most expensive diamonds

#Depth
sns.kdeplot(df['depth'], shade=True , color='r')
sns.scatterplot(x='depth',y ='price',data=df )
#The price varies considerably for the same range of depth

#Table
sns.kdeplot(df['table'], shade=True , color='r')
sns.scatterplot(x='table',y ='price',data=df )
#The price varies considerably for the same range of table

sns.catplot(x='x',data=df , kind='box')
sns.catplot(x='y',data=df , kind='box')
sns.catplot(x='z',data=df , kind='box')

"""Creating a new feature volume to combine length,width and height and droppng x,y,z"""
df['volume'] = df['x']*df['y']*df['z']
df=df.drop(['x','y','z'], axis=1)

"""Converting categorical data into numerical data for modelling"""
#For cut we need to preserve order
df['cut1'] = df['cut'].apply(lambda x: ['Ideal', 'Premium', 'Very Good', 'Good','Fair'].index(x))
df.dtypes
#Using onehotenocder for remaining categorical variables
ct=ColumnTransformer([('one-hot-encoder',OneHotEncoder(categories='auto'),[2,3])],remainder='passthrough')
df=ct.fit_transform(df)
df=pd.DataFrame(df)
df.drop([16],axis=1,inplace=True)

"""Splitiing dataset into train and test"""
#Separting target variable and independent variables"""
X = df.drop([19], axis=1)
y = df[19]

#Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=66)

"""Scaling data"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

""" Linear Regression """
lr = LinearRegression()
lr.fit(X_train , y_train)
accuracies = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 5,verbose = 1)
print(accuracies)
#[0.90845702 0.91729169 0.9139617  0.91247948 0.91806015]
y_pred_train = lr.predict(X_train)
mse = mean_squared_error(y_train, y_pred_train)
mae = mean_absolute_error(y_train, y_pred_train)
rmse = mean_squared_error(y_train, y_pred_train)**0.5
r2 = r2_score(y_train, y_pred_train)
print('MSE    : ' ,mse)
print('MAE    : ' ,mae)
print('RMSE   : ' ,rmse)
print('R2     : ' ,r2)
"""
MSE    :  1344562.2735258597
MAE    :  803.7156987666175
RMSE   :  1159.5526178340765
R2     :  0.9170634806060713

"""

y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('MSE    : ' ,mse)
print('MAE    : ' ,mae)
print('RMSE   : ' ,rmse)
print('R2     : ' ,r2)

"""
MSE    :  1336993.1424322904
MAE    :  802.9244144491277
RMSE   :  1156.2841962217983
R2     :  0.9170634806060713
"""# R2 score of 91.70 % is achieved.

"""Random Forest"""
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train , y_train)
accuracies = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 5,verbose = 1)
print(accuracies)
#[0.9791194  0.98070768 0.98196531 0.97774786 0.98100536]
y_pred_train = rf.predict(X_train)
mse = mean_squared_error(y_train, y_pred_train)
mae = mean_absolute_error(y_train, y_pred_train)
rmse = mean_squared_error(y_train, y_pred_train)**0.5
r2 = r2_score(y_train, y_pred_train)
print('MSE    : ' ,mse)
print('MAE    : ' ,mae)
print('RMSE   : ' ,rmse)
print('R2     : ' ,r2)
"""
MSE    :  42722.01485026903
MAE    :  108.82421369060496
RMSE   :  206.6930449973318
R2     :  0.997303272585458

"""

y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('MSE    : ' ,mse)
print('MAE    : ' ,mae)
print('RMSE   : ' ,rmse)
print('R2     : ' ,r2)

"""
MSE    :  285614.5983379929
MAE    :  271.6150016952992
RMSE   :  534.4292266876812
R2     :  0.9822827208887889

"""
# R2 score of 98.22 % is achieved.

