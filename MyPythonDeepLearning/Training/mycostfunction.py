'''
Created on 2017. 4. 10.

@author: Byoungho Kang
'''

import numpy as np

def meanSquaredError(y, t):
    return 0.5*np.sum((y-t)**2)

def crossEntropyError(y, t):
    print(y)
    y = y.reshape(1, y.size)
    t = t.reshape(1, t.size)
    print(y)
    delta = 1e-7 #아주 작은 값 (y가 0인 경우 -inf 값을 예방)
    return -np.sum(t*np.log(y+delta)) / y.shape[0]

t = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) # label = 5
y = np.array([0.1, 0.03, 0.05, 0.2, 0.9, 0.0, 0.1, 0.2, 0.12, 0.03]) # 5라고 추정
# 정답일 경우 MSE, CEE의 값은 적다.
print("-- 정답인 경우 --")
print("MSE :", meanSquaredError(y, t)) 
print("CEE :", crossEntropyError(y, t))

y = np.array([0.1, 0.03, 0.05, 0.2, 0.0, 0.1, 0.2, 0.12, 0.03, 0.9]) # 9라고 추정
# 오류일 경우 MSE, CEE의 값은 적다.
print("-- 오류인 경우 --")
print("MSE :", meanSquaredError(y, t))
print("CEE :", crossEntropyError(y, t))
