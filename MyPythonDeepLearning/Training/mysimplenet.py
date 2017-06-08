'''
Created on 2017. 4. 12.

@author: Byoungho Kang
'''

import numpy as np
from Common.functions import softmax, cross_entropy_error
from Common.gradient import numerical_gradient

class SimpleNet:
    
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 가중치 초기화 (2,3) 
        
    def predict(self, x):
        z = np.dot(x, self.W)
        y = softmax(z)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)
        return loss
    
simpleNet = SimpleNet()
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
print("input values:", x)
print("initialized weight:\n", simpleNet.W)
print("labeled values:", t)
print("neuralnet output:", simpleNet.predict(x))
print("cost(loss) :", simpleNet.loss(x, t))

def f(w): #w는 dummy
    return simpleNet.loss(x, t)
'''
 numerical_gradient(f,x)에서
 f는 손실함수, x는 손실함수 f의 인수
 즉, simpleNet.W[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]
 의 값으로 손실함수를 편미분한 결과 (Gradient)
'''
gradient = numerical_gradient(f, simpleNet.W)
print("gradient:\n", gradient)