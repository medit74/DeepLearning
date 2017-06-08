'''
Created on 2017. 4. 11.

@author: Byoungho Kang
'''

import numpy as np
import matplotlib.pyplot as plt
from Common.gradient_2d import numerical_gradient

'''
f = x[0]**2 + x[1]**2
initX = 초기값 (-3,4)
초기값에서 함수에 대한 기울기 (편미분에 의한 계산)를 구한 후
기울기 값과 학습률을 이용해서 x값을 이동
위를 stepNum(100번) 반복
'''
def gradientDescent(f, initX, learningRate, stepNum=100):
    x = initX
    xHistory = []
    
    for i in range(stepNum):
        xHistory.append(x.copy())
        grad = numerical_gradient(f,x)
        x -= learningRate*grad
        if(i == stepNum-1):
            print(x)
        
    return x, np.array(xHistory)

def sampleFunction(x):
    return x[0]**2 + x[1]**2

# 적정학습률
initX = np.array([-3.0,4.0])
learningRate = 0.1
x, xHistory = gradientDescent(sampleFunction, initX, learningRate)

# 학습률이 큰 경우
initX = np.array([-3.0,4.0])
learningRate = 2
x, xHistory = gradientDescent(sampleFunction, initX, learningRate)

# 학습률이 작은 경우
initX = np.array([-3.0,4.0])
learningRate = 0.001
x, xHistory = gradientDescent(sampleFunction, initX, learningRate)

# 그래프로 표현
plt.plot([-5,5],[0,0],'--b')
plt.plot([0,0],[-5,5],'--b')
plt.plot(xHistory[:,0],xHistory[:,1],'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
