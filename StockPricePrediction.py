## Imports all modules called within application.
import quandl
import math
import datetime
import pickle

import numpy as np
import pandas as pd

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
## Sets style of graphs being plotted
from matplotlib import style
style.use('ggplot')

## Downloads specified data (Google stock data) from Quand1
df = quandl.get("WIKI/GOOGL")
# print(df.head())

## Pairs down dataframe
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
## Creates a new % spread column that is based on the closing price
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
## Creates a new daily percent change column that is based on the closing price 
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
## Defines the new dataframe
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
# print(df.head())

## Defines a new column and handles missing data by replacing NaN cells with -99999
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

## Define features and labels
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
y = np.array(df['label'])

## Drops all NaN information from dataset
df.dropna(inplace=True)

##  Trains set of features, testing set of features, training set of labels, and testing set of labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
## Trains the clasifier
# clf = LinearRegression(n_jobs=-1)
# confidence = clf.score(X_test, y_test)
# print(confidence)

## Calls saved clasifier (after defining, training and testing) to speed up the training process
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

## Predicts Clasifier and adds 'Forcast' column
forecast_set = clf.predict(X_lately)
# print(forecast_set, confidence, forecast_out)
df['Forecast'] = np.nan

## Finds last day in the dataframe and assigns each new forecast to a new day
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

## Adds forcast to dataframe
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

## Graphs stock data and plots the forcast / prediction to the graph.
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
