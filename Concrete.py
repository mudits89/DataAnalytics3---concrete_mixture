# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:59:42 2018

@author: mudit
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
path = "C:/Users/mudit/Desktop/babel/Data Analytics/EDA/Construction Dataset/"
dataset = pd.read_csv(path + 'Concrete.csv')

#check for null values 
null_value_check = dataset.isnull().any()

#histogram plot
dataset['cement'].mean()
plt.hist(dataset.cement)
plt.savefig(path + 'cement_hist.png')

#correlation plot
plt.matshow(dataset.corr())
dataset.corr()
plt.savefig(path+'data_corr.png')
plt.show()


#correlation
dataset_corr = dataset.corr()

plt.scatter(dataset['cement'],dataset['csMPa'])
plt.savefig(path + 'scatter_cement_csMPa.png')

plt.scatter(dataset['slag'],dataset['csMPa'])
plt.savefig(path + 'scatter_slag_csMPa.png')

plt.scatter(dataset['flyash'],dataset['csMPa'])
plt.savefig(path + 'scatter_flyash_csMPa.png')

plt.scatter(dataset['age'],dataset['csMPa'])
plt.savefig(path + 'scatter_age_csMPa.png')

plt.scatter(dataset['water'],dataset['csMPa'])
plt.savefig(path + 'scatter_water_csMPa.png')

plt.scatter(dataset['superplasticizer'],dataset['csMPa'])
plt.savefig(path + 'scatter_superplasticizer_csMPa.png')

plt.scatter(dataset['coarseaggregate'],dataset['csMPa'])
plt.savefig(path + 'scatter_coarseaggregate_csMPa.png')

plt.scatter(dataset['fineaggregate'],dataset['csMPa'])
plt.savefig(path + 'scatter_fineaggregate_csMPa.png')


#split data into X and Y
X = dataset.iloc[:,0:-1]
Y = dataset.iloc[:,-1]

#Split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 17)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

#Using Validation Parameters
from sklearn.metrics import r2_score, mean_squared_error
print(r2_score(Y_test,Y_pred))
print(mean_squared_error(Y_test,Y_pred))