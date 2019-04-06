import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class load_data(object):

    def __init__(self):
        self.train = pd.read_csv('data/train_small.csv').fillna(0)
        self.split_data()

    def split_data(self):
        self.train = self.train.drop('Page', axis=1)
        row = self.train.iloc[6000, :].values
        X = row[0:549]
        y = row[1:550]

        # Splitting the dataset into the Training set and Test set

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # Feature Scaling

        sc = MinMaxScaler()
        X_train = np.reshape(X_train, (-1, 1))
        y_train = np.reshape(y_train, (-1, 1))
        self.X_train = sc.fit_transform(X_train)
        self.y_train = sc.fit_transform(y_train)
        self.X_train = np.reshape(self.X_train, (384, 1, 1))