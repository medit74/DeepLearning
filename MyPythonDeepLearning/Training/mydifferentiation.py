'''
Created on 2017. 4. 11.

@author: Byoungho Kang
'''

import numpy as np
import matplotlib.pyplot as plt

def numericalDiff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h)-f(x-h))/(2*h)

def sampleFunc1(x):
    return 0.01*x**2 + 0.1*x

def numeiralGradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) #x와 형상이 같은 배열을 생성
    for idx in range(x.size):
        tmpVal = x[idx]
        x[idx] = tmpVal + h
        print(idx, x)
        fxh1 = f(x)
        
        x[idx] = tmpVal - h
        print(idx, x)
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmpVal # 원래 값 복원
    
    return grad
    
'''
f(x) = x[0]**2 + x[1]**2
x[0] = 3, x[1] = 4일 때 편미분 구하기 위한 함수
'''
def sampleFunc2(x):
    return x**2 + 4**2
def sampleFunc3(x):
    return 3**2 + x**2    

'''
f(x0, x1) = x0**2 + x1**2인 함수
'''
def sampleFunc4(x):
    return x[0]**2 + x[1]**2

x = np.arange(0, 20, 0.1)
y = sampleFunc1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numericalDiff(sampleFunc1, 5))
print(numericalDiff(sampleFunc1, 10))

print(numericalDiff(sampleFunc2, 3.0))
print(numericalDiff(sampleFunc3, 4.0))

print(numeiralGradient(sampleFunc4, np.array([3.0, 4.0])))
print(numeiralGradient(sampleFunc4, np.array([0.0, 2.0])))
print(numeiralGradient(sampleFunc4, np.array([3.0, 0.0])))