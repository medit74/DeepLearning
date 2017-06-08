'''
Created on 2017. 4. 6.

@author: Byoungho Kang
'''

import numpy as np

'''
    3층 신경망의 가중치, 편항 초기화
'''
def initNeuralNetwork():
    neuralNetwork = {} # Python Dictonary 자료형
    neuralNetwork['W1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])       # 가중치1 : 2x3 shape
    neuralNetwork['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])    # 가중치2 : 3x2 shape
    neuralNetwork['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])                # 가중치3 : 2x2 shape
    neuralNetwork['b1'] = np.array([0.7, 0.8, 0.9]) # Bias1 : 3차원 Vector
    neuralNetwork['b2'] = np.array([0.7, 0.8])      # Bias2 : 2차원 Vector
    neuralNetwork['b3'] = np.array([0.7, 0.8])      # Bias3 : 2차원 Vector
    
    return neuralNetwork

'''
    은닉층의 활성화 함수로 Sigmoid함수를 사용
'''
def sigmoidFunction(x):
    return 1/(1+np.exp(-x))

'''
    출력층의 활성화 함수로 항등함수 사용
'''
def identityFunction(x):
    return x

'''
    입력값을 FeedForward하여 출력값을 가져온다
    ia (input numpy array)는 2차원 Vector
'''
def feedForward(neuralNetwork, ia):
    # 신경망 초기값을 가져온다.
    W1, W2, W3 = neuralNetwork['W1'], neuralNetwork['W2'], neuralNetwork['W3']
    b1, b2, b3 = neuralNetwork['b1'], neuralNetwork['b2'], neuralNetwork['b3']
    # 첫번째 은닉층의 계산
    z1 = sigmoidFunction(np.dot(ia, W1) + b1) # 입력(2차원벡터) x 가중치(2x3 행렬) + 3차원벡터
    print("1st 은닉층 값 : " + str(z1))
    # 두번째 은닉층의 계산
    z2 = sigmoidFunction(np.dot(z1, W2) + b2) # 입력(2차원벡터) x 가중치(2x3 행렬) + 3차원벡터
    print("2nd 은닉층 값 : " + str(z2))
    # 출력층의 계산
    y = identityFunction(np.dot(z2, W3) + b3) # 입력(2차원벡터) x 가중치(2x3 행렬) + 3차원벡터
    return y
    
'''
    신경망 계산
'''    
neuralNetwork = initNeuralNetwork()
ia = np.array([4.5, 6.2])
print("입력층 값 : " + str(ia))
oa = feedForward(neuralNetwork, ia)
print("출력층 값 : " + str(oa))