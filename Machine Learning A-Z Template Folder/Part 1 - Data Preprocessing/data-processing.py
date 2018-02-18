# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read in data
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
x[:, 0] = labelEncoderX.fit_transform(x[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
x = oneHotEncoder.fit_transform(x).toarray()
labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)

# split dataset into a training set and a test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# feature scaling to put values on the same scale
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
