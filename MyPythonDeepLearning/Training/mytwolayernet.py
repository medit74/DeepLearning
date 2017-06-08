'''
Created on 2017. 4. 12.

@author: Byoungho Kang
'''

import numpy as np
import Common.functions as cf
from Common.gradient import numerical_gradient

class MyTwoLayerNet:
    
    def __init__(self, inputSize, hiddenSize, outputSize, weightInitStd = 0.01):
        self.params = {}
        self.params['W1'] = weightInitStd * np.random.randn(inputSize, hiddenSize)
        self.params['W2'] = weightInitStd * np.random.randn(hiddenSize, outputSize)
        self.params['b1'] = np.zeros(hiddenSize)
        self.params['b2'] = np.zeros(outputSize)
        
    def predict(self, x):
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        z = cf.sigmoid(np.dot(x, W1) + b1)
        y = cf.softmax(np.dot(z, W2) + b2)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cf.cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
       
    ''' Gradient 구하기 (수치 미분을 이용) '''
    def numericalGradient(self, x, t):
        lossW = lambda W : self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(lossW, self.params['W1'])
        grads['W2'] = numerical_gradient(lossW, self.params['W2'])
        grads['b1'] = numerical_gradient(lossW, self.params['b1'])
        grads['b2'] = numerical_gradient(lossW, self.params['b2'])
        return grads
        
        
        