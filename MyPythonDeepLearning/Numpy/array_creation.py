'''
Created on 2017. 4. 7.
@author: Byoungho Kang
'''

import numpy as np

'''
ndarray creation
create an array from a regular python list
ndarray.ndim     : number of dimensions
ndarray.shape    : the size of the array in each dimension
ndarray.size     : total number of elements
ndarray.dtype    : the type of the elements 
ndarray.itemsize : the size in bytes of each element
'''
a1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([[0, 1, 2],[3, 4, 5]], dtype="float32")

print(a1, type(a1))
print(a2, type(a2))
print(a1.ndim, a1.shape, a1.size, a1.dtype, a1.itemsize)
print(a2.ndim, a2.shape, a2.size, a2.dtype, a2.itemsize)
print("---------------------------------------------------")

'''
ndarray creation
numpy.zeros(shape[, dtype, order])
numpy.ones((shape[, dtype, order])
numpy.empty(shape, dtype=float, order='C') #random
numpy.random.rand(d0, d1, ..., dn) # (0, 1) 사이의 임의의 부동소수점 수 생성 (균일분포, uniform distribution)
numpy.random.rand(d0, d1, ..., dn) # 평균 0, 표준편차 1의 표준 정규분포(standard normal distribution)에 따르는 임의의 부동소수점 수 생성
numpy.random.choice(a, size=None, replace=True, p=None) Generates a random sample from a given 1-D array
'''
a1 = np.zeros((3,4))
a2 = np.ones((2,3,4), dtype=np.int16)
a3 = np.empty((2,3), dtype="complex")
a4 = np.random.rand(3,2)
a5 = np.random.randn(3,2)
a6 = np.random.choice(100,5)  #0~100사이 중 5개 선택
print(a1, a1.ndim, a1.shape, a1.size, a1.dtype)
print(a2, a2.ndim, a2.shape, a2.size, a2.dtype)
print(a3, a3.ndim, a3.shape, a3.size, a3.dtype)
print(a4, a4.ndim, a4.shape, a4.size, a4.dtype)
print(a5, a5.ndim, a5.shape, a5.size, a5.dtype)
print(a6, a6.ndim, a6.shape, a6.size, a6.dtype)
print("---------------------------------------------------")

'''
ndarray creation (이미 생성된 배열과 같은 형태의 배열)
numpy.zeros_like(a, dtype=None, order='K', subok=True)
numpy.ones_like(a, dtype=None, order='K', subok=True)
numpy.empty_like(a, dtype=None, order='K', subok=True)
'''
a1 = np.array([[0, 1, 2],[3, 4, 5]], dtype="float32")
a2 = np.zeros_like(a1)
a3 = np.ones_like(a1)
a4 = np.empty_like(a1)
print(a1, a1.ndim, a1.shape, a1.size, a1.dtype)
print(a2, a2.ndim, a2.shape, a2.size, a2.dtype)
print(a3, a3.ndim, a3.shape, a3.size, a3.dtype)
print(a4, a4.ndim, a4.shape, a4.size, a4.dtype)
print("---------------------------------------------------")

'''
ndarray creation
numpy.arange([start,]stop,[step,]dtype=None)
'''
a1 = np.arange(15)
a2 = np.arange(-5, 5, 0.5)
print(a1, a1.ndim, a1.shape, a1.size, a1.dtype)
print(a2, a2.ndim, a2.shape, a2.size, a2.dtype)
print("---------------------------------------------------")

'''
ndarray conversion
numpy.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
'''
a1 = np.array([1, 2, 2.5])
a2 = a1.astype("int32")
print(a1, a1.ndim, a1.shape, a1.size, a1.dtype)
print(a2, a2.ndim, a2.shape, a2.size, a2.dtype)
print("---------------------------------------------------")

'''
shape manipulation
numpy.reshape
numpy.flatten
'''
a1 = np.arange(15)
a2 = a1.reshape(3,5)
a3 = a1.reshape(1,3,5)
a4 = a3.flatten()
print(a1, a1.ndim, a1.shape, a1.size, a1.dtype)
print(a2, a2.ndim, a2.shape, a2.size, a2.dtype)
print(a3, a3.ndim, a3.shape, a3.size, a3.dtype)
print(a4, a4.ndim, a4.shape, a4.size, a4.dtype)
print("---------------------------------------------------")
