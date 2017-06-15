'''
Created on 2017. 5. 12.
@author: Byoungho Kang
'''

import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, TimeDistributed
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

'''
Training Set
'''
mySample = "HelloMyKerasWorld"
mySampleCharSet = set(mySample)
mySampleCharSetDict = {char: idx for idx, char in enumerate(mySampleCharSet)}
print(mySample, mySampleCharSet, mySampleCharSetDict)

X_str = mySample[:-1]
Y_str = mySample[1:]
X = [mySampleCharSetDict[char] for char in X_str]
Y = [mySampleCharSetDict[char] for char in Y_str]
print(X_str, X, Y_str, Y)

'''
Hyper Parameter
'''
data_dim = len(mySampleCharSet)
timesteps = len(Y_str)
num_classes = len(mySampleCharSet)
print(data_dim, timesteps, num_classes)

# one-hot encoding
X = to_categorical(X, data_dim)
X = np.reshape(X, (-1, len(X), data_dim))
Y = to_categorical(Y, data_dim)
Y = np.reshape(Y, (-1, len(Y), data_dim))
print(X, X.shape)
print(Y, Y.shape)

'''
Build a Model
'''
y = Sequential()
y.add(LSTM(num_classes, activation='tanh', input_shape=(timesteps, data_dim), return_sequences=True))
y.add(TimeDistributed(Dense(num_classes)))
y.add(Activation('softmax'))
y.summary()

'''
Compile a Model
'''
y.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

'''
Training a Model
'''
y.fit(X, Y, epochs=500)

'''
Predict
'''
predictions = y.predict(X, verbose=0)

for i, prediction in enumerate(predictions):
    print(prediction)
    
    x_index = np.argmax(X[i], axis=1)
    x_str = [list(mySampleCharSet)[j] for j in x_index]
    print(x_index, ''.join(x_str))
    
    index = np.argmax(prediction, axis=1)
    result = [list(mySampleCharSet)[j] for j in index]
    print(index, ''.join(result))
