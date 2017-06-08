'''
Created on 2017. 4. 7.
@author: Byoungho Kang
'''

import numpy as np

a = np.arange(0, 6)
print(a)
print(np.sin(a))
print(np.cos(a))
print(np.tanh(a))
print(np.exp(a))
print("---------------------------------------------------")

a = np.arange(0.01, 1, 0.05)
print(a)
print(np.log(a))
print("---------------------------------------------------")

a = np.arange(0, 6)**2
print(a)
print(np.sqrt(a))
print("---------------------------------------------------")

a1 = np.array([2,3,4])
a2 = np.array([1,5,2])
print(np.maximum(a1,a2))
print(np.minimum(a1,a2))
print(np.sum(a1))
print("---------------------------------------------------")

'''
numpy.argmax() : Returns the indices of the maximum values
'''
a = np.array([6,2,3,1,4,5])
print(a)
print(np.argmax(a), a[np.argmax(a)])
a = np.array([[0.1,0.8,0.2],[0.3,0.2,0.5],[0.9,0.5,0.3]])
print(np.argmax(a, axis=1))