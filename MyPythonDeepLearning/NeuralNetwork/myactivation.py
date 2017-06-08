'''
Created on 2017. 4. 6.

@author: Byoungho Kang
'''

import numpy as np
import matplotlib.pyplot as plt

'''
    계단함수 구현
    입력파라미터 numpy.ndarray
    출력파라미터 numpy.ndarray
'''
def stepFunction(x):
    y = [] # Empty List
    for value in x:
        if value > 0:
            y.append(1)
        else:
            y.append(0)
        
    return np.array(y)

'''
    시그모이드함수 구현
    입력파라미터 numpy.ndarray
    출력파라미터 numpy.ndarray
    np.exp(-x) : x가 numpy.ndarray type이라도 broadcast 되어 연산됨.
'''
def sigmoidFunction(x):
    return 1/(1+np.exp(-x))

'''
    Rectified Linear Unit 함수 구현
    입력파라미터 numpy.ndarray
    출력파라미터 numpy.ndarray
'''
def reluFunction(x):
    return np.maximum(0, x)

'''
    Softmax함수 구현
    입력파라미터 numpy.ndarray
    출력파라미터 numpy.ndarray
'''
def softmaxFunction(x):
    expX = np.exp(x - np.max(x))    # Overflow 대비를 위해 원소 중 최대값을 각 입력 신호에서 빼서 예방한다.
    sumExpX = np.sum(expX)          # 지수 함수 결과의 합
    return expX / sumExpX   
        
x = np.arange(-5, 5, 0.01)
y1 = stepFunction(x)
y2 = sigmoidFunction(x)
y3 = reluFunction(x)

a = np.array([2.3, -0.9, 3.6])
y4 = softmaxFunction(a)
print(y4, np.sum(y4))

a = np.array([900, 1000, 1000])
y4 = softmaxFunction(a)
print(y4, np.sum(y4))

plt.plot(x, y1, label="Step Function"   ,linestyle="-")
plt.plot(x, y2, label="Sigmoid Function",linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Step & Sigmoid")
plt.legend()
plt.show()

plt.plot(x, y3, label="ReLU Function"   ,linestyle="-.")
plt.xlabel("x")
plt.ylabel("y")
plt.title("ReLU Function")
plt.legend()
plt.show()