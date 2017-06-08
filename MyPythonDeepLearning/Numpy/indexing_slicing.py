'''
Created on 2017. 4. 7.
@author: Byoungho Kang
'''

import numpy as np

'''
ndarray creation
'''
a1 = np.arange(10)**2
a2 = np.arange(1,21,1).reshape(4,5)
print(a1, a1.ndim, a1.shape, a1.size, a1.dtype)
print(a2, a2.ndim, a2.shape, a2.size, a2.dtype)
print("---------------------------------------------------")

'''
ndarray indexing & slicing
'''
print(a1[2])
print(a1[3:5])
print(a1[0:6:2]) #0에서 6번까지 매2번째 요소마다(6번째는 포함 안됨)
print(a1[::-1]) #reversed a
print(a2[2,3])
print(a2[0:4,1])
print(a2[2:4])
print(a2[1])
print(a2[-1])
print("---------------------------------------------------")

'''
ndarray iterating
'''
for element in a1:
    print(element+1)
print("---------------------------------------------------")

'''
ndarray slicing은 복사본을 생성하지 않고 메모리 영역을 반환한다.
'''
a = np.ones(4)
print(a)
view = a[0:2]
view[0] = 2
print(a)