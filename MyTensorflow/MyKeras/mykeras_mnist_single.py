'''
Created on 2017. 5. 12.
@author: Byoungho Kang
'''

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

'''
Training Set
'''
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
Y_train = np_utils.to_categorical(Y_train)
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
Y_test = np_utils.to_categorical(Y_test)

'''
Build a Model
'''
y = Sequential()
y.add(Dense(units=64, input_dim=28*28, activation='relu'))
y.add(Dense(units=10, activation='softmax'))

'''
Compile a Model
'''
y.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

'''
Training a Model
'''
y.fit(X_train, Y_train, batch_size=32, epochs=5)

'''
Evaluate a Model
'''
loss_and_metrics = y.evaluate(X_test, Y_test, batch_size=32)

print('\nloss_and_metrics : ' + str(loss_and_metrics))