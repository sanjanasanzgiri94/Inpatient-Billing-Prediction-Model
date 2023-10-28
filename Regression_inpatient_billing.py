#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:11:08 2023

@author: susanzgiri
"""

###Regression part

import numpy as np
import pandas as pd

d1 = pd.read_csv(r"/Users/susanzgiri/Downloads/Inpatientbill.csv")
print(d1)

d1.info()
d1.isnull().sum()
d1.dropna(inplace = True)
d1.duplicated().sum()
d1
d1.info()

m = d1.loc[:,[' Average Covered Charges ',' Average Total Payments ', 'Average Medicare Payments']]
m

n = d1.loc[:, ['Provider State']]
x = d1.loc[:,[' Average Covered Charges ',' Average Total Payments ']]
x
y = d1.loc[:,['Average Medicare Payments']]
y
#regression method

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=5)
#x_train - input parameter,x_test - kept to ourself for testing
#70% data used for training and 30% data for testing

x_train
x_test
y_train
y_test


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(n)
numerical_labels = label_encoder.transform(n)
d1['location_labels'] = numerical_labels
d1.columns
d1[["location_labels", "Provider State"]]

x = d1.loc[:,[' Average Covered Charges ',' Average Total Payments ', 'location_labels']]
x
y = d1.loc[:,['Average Medicare Payments']]
y


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

reg.fit(x,y)
reg.predict([[329000, 5800, 34]])
reg.predict([[49088, 3800, 2]])
reg.predict(x)


print(reg.score(x,y))
reg.score(x_train, y_train)         ##????
reg.score(x_test, y_test)               ##???
