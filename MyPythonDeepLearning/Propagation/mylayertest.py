'''
Created on 2017. 4. 13.

@author: Byoungho Kang
'''

import numpy as np
import Common.layers as layers

x = np.array([[1.1, -0.2],[-1.5, 2.7]])
difOut = np.array([[1.0, 1.0],[1.0, 1.0]])
relu = layers.Relu()
print("ReLU : input")
print(x)
print("ReLU : forward")
print(relu.forward(x))
print("ReLU : dif. out")
print(difOut)
print("ReLU : backward")
print(relu.backward(difOut))

x = np.array([[1.1, -0.2],[-1.5, 2.7]])
difOut = np.array([[1.0, 1.0],[1.0, 1.0]])
sigmoid = layers.Sigmoid()
print("Sigmoid : input")
print(x)
print("Sigmoid : forward")
print(sigmoid.forward(x))
print("Sigmoid : dif. out")
print(difOut)
print("Sigmoid : backward")
print(sigmoid.backward(difOut))
