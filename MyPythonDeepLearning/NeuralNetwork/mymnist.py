'''
Created on 2017. 4. 7.

@author: Byoungho Kang
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
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

'''
load dataset and initialize weight & bias
'''
(trainImg, trainLbl),(testImg, testLbl) = getData()
network = initNetwork();

'''
show first element
'''
firstImg = trainImg[0].reshape(28,28)
firstLbl = trainLbl[0]
print(firstLbl) # 5
plt.imshow(firstImg, cmap='gray', interpolation='nearest')
plt.show()

'''
predict
'''
batchSize = 100
accuracyCnt = 0
print(len(trainImg)) # 60000
for idx in range(0, len(trainImg), batchSize):
    y = predict(network, trainImg[idx:idx+batchSize])
    p = np.argmax(y, axis=1)
    accuracyCnt += np.sum(p == trainLbl[idx:idx+batchSize])

print(accuracyCnt)
print("Accuracy:" + str(accuracyCnt / len(trainImg)))

