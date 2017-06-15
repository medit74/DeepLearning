'''
Created on 2017. 5. 12.
@author: Byoungho Kang
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler #scikit-learn (http://scikit-learn.org/stable/)
from sklearn.metrics import mean_squared_error

'''
Training Set
'''
rawdata = pd.read_csv("../resources/international-airline-passengers.csv", usecols = [1])
print(rawdata.head())
print(rawdata.values, rawdata.values.dtype)
plt.plot(rawdata)
plt.show()

scaler = MinMaxScaler(feature_range = (0,1))
dataset = scaler.fit_transform(rawdata.values[0:-1])
print(dataset)

train_size = int(len(dataset) * 0.7)
test_size  = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset),:]
print(len(dataset), len(train), len(test))

def create_dataset(dataset, look_back = 1):
    """
    - look_back: number of previous time steps
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i+look_back, 0])
    return np.array(X), np.array(Y)
look_back = 1
train_X, train_y = create_dataset(train, look_back)
test_X,  test_y  = create_dataset(test,  look_back)
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X  = np.reshape(test_X,  (test_X.shape[0],  1, test_X.shape[1]))
print(train_X, train_y)
print(test_X, test_y)

'''
Hyper Parameter
'''

'''
Build a Model
'''
model = Sequential()
model.add(LSTM(4, input_dim = look_back))
model.add(Dense(1))

'''
Compile a Model
'''
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

'''
Training a Model
'''
model.fit(train_X, train_y, nb_epoch = 100, batch_size = 1, verbose = 2)

'''
Predict
'''
train_pred = model.predict(train_X)
test_pred  = model.predict(test_X)
train_pred = scaler.inverse_transform(train_pred)
train_y    = scaler.inverse_transform([train_y])
test_pred  = scaler.inverse_transform(test_pred)
test_y     = scaler.inverse_transform([test_y])

'''
Accuracy
'''
train_score = math.sqrt(mean_squared_error(train_y[0], train_pred[:,0]))
test_score  = math.sqrt(mean_squared_error(test_y[0],  test_pred[:,0]))
train_pred_plot = np.empty_like(dataset)
train_pred_plot[:,:] = np.nan
train_pred_plot[look_back:len(train_pred)+look_back, :] = train_pred

test_pred_plot = np.empty_like(dataset)
test_pred_plot[:, :] = np.nan
test_pred_plot[len(train_pred)+(look_back*2)+1:len(dataset)-1, :] = test_pred

plt.plot(scaler.inverse_transform(dataset))
plt.plot(train_pred_plot)
plt.plot(test_pred_plot)
plt.show()