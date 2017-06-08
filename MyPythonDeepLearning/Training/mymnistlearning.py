'''
Created on 2017. 4. 12.

@author: Byoungho Kang
'''

import numpy as np
import matplotlib.pyplot as plt
from Common.mnist import load_mnist
from Training.mytwolayernet import MyTwoLayerNet

(trainImg, trainLbl),(testImg, testLbl) = load_mnist(one_hot_label=True)
network = MyTwoLayerNet(784, 50, 10)

# hyper parameters
itersNum = 1000 # 반복횟수
trainSize = trainImg.shape[0] # 60000
batchSize = 100 # mini-bach 크기
learningRate = 0.1 # 학습률

# 누적기록
trainLossList = []

print("-- Start Learning -- ")

for i in range(itersNum):
    # mini-batch 획득
    miniBatchMask = np.random.choice(trainSize, batchSize)
    trainImgBatch = trainImg[miniBatchMask]
    trainLblBatch = trainLbl[miniBatchMask]
        
    # Gradient 계산
    grad = network.numericalGradient(trainImgBatch, trainLblBatch)
    
    # 가중치, 편향 갱신
    for key in ('W1', 'W2', 'b1', 'b2'):
        network.params[key] -= learningRate*grad[key]
        
    # 비용함수(오차)의 변화 기록
    loss = network.loss(trainImgBatch, trainLblBatch)
    trainLossList.append(loss)
    
    print("iteration", i, ":", loss)

print("-- End Learning -- ")    
    
# 그래프 그리기
x = np.arange(len(trainLossList))
plt.plot(x, trainLossList, label="loss")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()
