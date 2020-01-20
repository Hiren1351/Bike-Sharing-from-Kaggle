# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:02:53 2020

@author: Hiru_Hunter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


bike = pd.read_csv('hour.csv')
bike.shape
bike.describe()
bike.columns

#We dont need some of the coulmn so we just remove it
#index,date,casual and egistered becaus it is total for demand
#first create a copy of dataframe

bikes = bike.copy()
bikes = bikes.drop(['index','date','casual','registered'],axis = 1)

#Pandas visualization for quick
bikes.hist(rwidth = 0.9)
plt.tight_layout()  

"""We can see that the predicted variable - Demand is not normally
Distributed"""

#Data Visualization with all continuous features vs Demand

plt.subplot(2,2,1)
plt.title('Temperature vs Demand')    
plt.scatter(bikes['temp'],bikes['demand'],s = 2,c = 'g') 

plt.subplot(2,2,2)
plt.title('Atemp vs Demand')
plt.scatter(bikes['atemp'],bikes['demand'],s = 2,c = 'b')

plt.subplot(2,2,3)
plt.title('Humidity vs Demand')
plt.scatter(bikes['humidity'],bikes['demand'],s = 2,c = 'c')

plt.subplot(2,2,4)
plt.title('WindSpeed vs Demand')
plt.scatter(bikes['windspeed'],bikes['demand'],s = 2,c = 'r')

plt.tight_layout()

#Data Visualization with all Categorical features vs Demand

colors = ['goldenrod','purple','tomato','orchid']
plt.subplot(3,3,1)
plt.title('Average Demand Per Season')
cat_list = bikes['season'].unique()#Create a list of all unique category for season
cat_aver = bikes.groupby('season').mean()['demand']#taking mean for demand of particular season
plt.bar(cat_list,cat_aver,color = colors)


plt.subplot(3,3,2)
plt.title('Average Demand Per Year')
year_list = bikes['year'].unique()#Create a list of all unique category for year
year_aver = bikes.groupby('year').mean()['demand']#taking mean for demand of particular year
plt.bar(year_list,year_aver,color = colors)

plt.subplot(3,3,3)
plt.title('Average Demand Per Month')
mo_list = bikes['month'].unique()#Create a list of all unique category for month
mo_aver = bikes.groupby('month').mean()['demand']#taking mean for demand of particular month
plt.bar(cat_list,cat_aver,color = colors)

plt.subplot(3,3,4)
plt.title('Average Demand Per hour')
hou_list = bikes['hour'].unique()#Create a list of all unique category for hour
hou_aver = bikes.groupby('hour').mean()['demand']#taking mean for demand of particular hour
plt.bar(hou_list,hou_aver,color = colors)

plt.subplot(3,3,5)
plt.title('Average Demand for Holiday')
ho_list = bikes['holiday'].unique()#Create a list of all unique category for Holiday
ho_aver = bikes.groupby('holiday').mean()['demand']#taking mean for demand of particular Holiday
plt.bar(ho_list,ho_aver,color = colors)

plt.subplot(3,3,6)
plt.title('Average Demand Per Week-Day')
week_list = bikes['weekday'].unique()#Create a list of all unique category for weekday
week_aver = bikes.groupby('weekday').mean()['demand']#taking mean for demand of particular weekday
plt.bar(week_list,week_aver,color = colors)

plt.subplot(3,3,7)
plt.title('Average Demand Per Working-Day')
wo_list = bikes['workingday'].unique()#Create a list of all unique category for month
wo_aver = bikes.groupby('workingday').mean()['demand']#taking mean for demand of particular month
plt.bar(wo_list,wo_aver,color = colors)

plt.subplot(3,3,8)
plt.title('Average Demand for Weather')
wea_list = bikes['weather'].unique()#Create a list of all unique category for weather
wea_aver = bikes.groupby('weather').mean()['demand']#taking mean for demand of particular weather
plt.bar(wea_list,wea_aver,color = colors)

plt.tight_layout()

"""We can see that the week day is normal,not influence the or not 
significant impact or no trend or pattern so we dont need it
Simillaly we have only 2 category in year so nu much impact and 
for working day there is no such significant its normal no pattern 
or trend so we also remove it."""

#Conclusion
"""1 - Demand is not normally distributed
   2 - Temperature and Demand appears to heve direct correlation
   3 - The plot of Temp and Atemp appear almost identical or simmilar.
   4 - Humidity and wind speed affect demand but need more statistical analysis.
   5 - There is a variation in demand based on "Season,Month,Holiday,Hour,Weather"
   6 - No Significance chance in demand due to "Week-Day,Working-Day"
   7 - Year wise growth pattern not considered due to limitted number of year."""
   
#Check for Outliers
   
bikes['demand'].describe()
bikes['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99])#Quartlie for % wise value of demand

#Test the Multicollinearity Assumpition

#Checking correlation coefficient matrix for linearity using corr
   
correlation = bikes[['temp','atemp','humidity','windspeed','demand']].corr()

"""As per result we can see that temp and atemp is highly correlated there is multicollinearity
so we check which one is more affect on demand so temp is more affect so we drop or remove atemp
simillaly wind speed has very low affect 0.09 so also drop it."""

#Want to Drop feature list
#weekday,year,workingday,atemp,windspeed

bikes = bikes.drop(['year','weekday','workingday','atemp','windspeed'],axis = 1)

#Checking Auto Correlation in Demand using acor
#we know that for checking autocorrelation we must convert int to float
df1 = pd.to_numeric(bikes['demand'],downcast = 'float')
plt.acorr(df1,maxlags = 12) #beacause we have 24 hour so we use maxlags as 12 hour time

#We can see that high autocorrelation for the "demand" feature.

#Now we know that demand or predicted feature is not normal so we convert it into log

df2 = bikes['demand']#Firs we chech if there is some effective result than we implement
df3 = np.log(df2)

plt.figure()
df2.hist(rwidth = 0.9,bins = 20)

plt.figure()
df3.hist(rwidth = 0.9,bins = 20)

#We can see that there is good output so we implement in our dataset

bikes['demand'] = np.log(bikes['demand'])

#Solving the problem of autocorrelation shift the demand by 3 lag

t_1 = bikes['demand'].shift(+1).to_frame()
t_1.columns =['t-1'] 

t_2 = bikes['demand'].shift(+2).to_frame()
t_2.columns =['t-2']

t_3 = bikes['demand'].shift(+3).to_frame()
t_3.columns =['t-3']

#now we concate to dataframe but avoid for overwrite we create new dataframe

bikes_new = pd.concat([bikes,t_1,t_2,t_3],axis = 1)

#now remove nan from dataset

bikes_new = bikes_new.dropna()

#Create dummy variable and drop first to avoiding dummy variable trap for categories

bikes_new.dtypes

"""we can see that some of feature type is int so for conversion of dummy variable
 we have toconvert its data type from int to categories"""
 
bikes_new['season'] = bikes_new['season'].astype('category')
bikes_new['month'] = bikes_new['month'].astype('category')
bikes_new['hour'] = bikes_new['hour'].astype('category')
bikes_new['holiday'] = bikes_new['holiday'].astype('category')
bikes_new['weather'] = bikes_new['weather'].astype('category')

bikes_new = pd.get_dummies(bikes_new,drop_first = True)

#Our demand variable is time series type of data so we can not split randomely if we do
#than time integraty is break.

Y = bikes_new[['demand']]
X = bikes_new.drop(['demand'],axis = 1)

train_size = 0.7 * len(X)
#we can see that train_size is float value so we have to convert it into int
train_size = int(train_size)

X_train = X.values[0: train_size]
X_test = X.values[train_size : len(X)]

Y_train = Y.values[0: train_size]
Y_test = Y.values[train_size : len(Y)]

#Create Ml Model and fit it

from sklearn.linear_model import LinearRegression
Lr = LinearRegression()
Lr.fit(X_train,Y_train)

Prediction = Lr.predict(X_test)

#Checking for Over fitting

R_Train = Lr.score(X_train,Y_train)
R_Test = Lr.score(X_test,Y_test)

#R-Score for Model
from sklearn import metrics
metrics.explained_variance_score(Y_test,Prediction)

#Three common evaluation metrics for regression problems:

MAE =metrics.mean_absolute_error(Y_test,Prediction) 
MAE

MSE = metrics.mean_squared_error(Y_test,Prediction)
MSE

RMSE = np.sqrt(metrics.mean_squared_error(Y_test,Prediction))
RMSE

#Model Summary

import statsmodels.api as sm
X2 = sm.add_constant(X_train)
model = sm.OLS(Y_train,X2).fit()
model.summary()

#Calculate RMSLE
#Prefered for non nagative predictions
#Less Variation for small and Large Prediction

#We know that our demand feature is converted in log so we have to convert
#back it again from log to number and exponent opposite to log

"""Equation of RMSLE in Words,

RMSLE = sqrt(sum(log(pi + 1) - log(ai + 1)**2)/n)

Where pi is predicted value and ai is actual value"""

Y_test_e = []
Prediction_e = []

for i in range(0,len(Y_test)):
    Y_test_e.append(np.exp(Y_test[i]))
    Prediction_e.append(np.exp(Prediction[i]))
    
Sum_rmsle = 0.0

for i in range(0,len(Y_test_e)):
    Log_ai = np.log(Y_test_e[i] + 1)
    Log_pi = np.log(Prediction_e[i] + 1)
    Squared_Diff = (Log_pi - Log_ai)**2
    Sum_rmsle = Sum_rmsle + Squared_Diff
    
    RMSLE = np.sqrt(Sum_rmsle/len(Y_test))

print(RMSLE)


#Create a scatterplot of the real test values versus the predicted values.

sns.scatterplot(Y_test_e,Prediction_e)

#Residual Histogram
    
sns.distplot((Y_test-Prediction),bins = 30)