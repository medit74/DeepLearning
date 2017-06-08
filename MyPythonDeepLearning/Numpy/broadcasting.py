'''
Created on 2017. 4. 7.
@author: Byoungho Kang
'''

import numpy as np

x1 = np.array([1.0,  2.0,  3.0])
y1 = np.array([5.0, 10.0, 15.0])
x2 = np.array([[1.0, 2.0],[ 3.0, 4.0]])
y2 = np.array([[5.0,10.0],[15.0,20.0]])
z1 = np.array([-1.0, -2.0])
z2 = np.array([[5.0],[10.0],[15.0]])

'''
ndarray basic operation
'''
print(x1 + y1)
print(x1 - y1)
print(x1 * y1)
print(x1 / y1)
print(x2 + y2)
print(x2 * y2)
print("---------------------------------------------------")

'''
ndarray broadcast
'''
print(x2 + z1)
print(x2 * z1)
print(x1 + z2)
print(x1**2)
print(x1>=2)