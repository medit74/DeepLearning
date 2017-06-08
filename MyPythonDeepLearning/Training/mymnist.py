'''
Created on 2017. 4. 7.

@author: Byoungho Kang
'''

import numpy as np
import pickle
from Common.mnist import load_mnist
from Common.functions import sigmoid, softmax

'''
load mnist data set
'''
def getData():
    
    (trainImg, trainLbl),(testImg, testLbl) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    
    print(trainImg.shape) # (60000, 784)
    print(trainLbl.shape) # (60000,)
    print(testImg.shape)  # (10000, 784)
    print(testLbl.shape)  # (10000,)

    return (trainImg, trainLbl),(testImg, testLbl)

def initNetwork():
    with open("../resources/sample_weight.pkl","rb") as f:
        network = pickle.load(f)
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        print(W1.shape, W2.shape, W3.shape)
        print(b1.shape, b2.shape, b3.shape)
    return network

def predict(network,x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)    
    return y

def crossEntropyError(y, t):
    # 1차원 배열인 경우 2차원 배열로 변경
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    miniBatchSize = y.shape[0]
    '''
    y[np.arange(miniBatchSize), t] 은 정답 레이블에 해당되는 신경망의 출력을 추출
    예) 만약 배치사이즈가 10개라면 아래와 같이 10개의 신경망 추출에 대한 출력값을 추출
       y[0,2], y[1,7], y[2,0], y[3,9],...y[9.5]
    '''
    return -np.sum(t*np.log(y[np.arange(miniBatchSize), t])) / miniBatchSize

'''
load dataset and initialize weight & bias
'''
(trainImg, trainLbl),(testImg, testLbl) = getData()
network = initNetwork();

'''
test first element
'''
img = trainImg[0].reshape(28,28) * 255
lbl = trainLbl[0]

'''
predict
 - MiniBatch Target Setting
 - Predict (추론 수행)
 - Cost Function (비용함수계산)
'''
miniBatchSize = 10
print(len(trainImg), "개 중에 ", miniBatchSize, "를 미니배치 학습한다.")
miniBatchMask = np.random.choice(len(trainImg), miniBatchSize)
trainImgBatch = trainImg[miniBatchMask]
trainLblBatch = trainLbl[miniBatchMask]
print("Random Choice 된 정답 레이블", trainLblBatch)

y = predict(network, trainImgBatch)
print("신경망 출력 결과 (10-배치크기, 10-분류크기) Shape\n", y)
p = np.argmax(y, axis=1)
print("최대값으로 추정한 결과", p)

print(crossEntropyError(y, trainLblBatch))
