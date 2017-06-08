'''
Created on 2017. 4. 6.

@author: Byoungho Kang
'''

import numpy as np

npArr1 = np.array([[1,2],[3,4]])        # 2x2 행렬 (2차원배열)
npArr2 = np.array([[5,6],[7,8]])        # 2x2 행렬 (2차원배열)
npArr3 = np.array([[1,2,3],[4,5,6]])    # 2x3 행렬 (2차원배열)
npArr4 = np.array([[5,6],[7,8],[9,10]]) # 3x2 행렬 (2차원배열)
npArr5 = np.array([1,2])                # 2x1 벡터 (1차원배열)

npResult1 = np.dot(npArr1, npArr2) # matrix inner product (2x2 행렬)
npResult2 = np.dot(npArr3, npArr4) # matrix inner product (2x2 행렬)
npResult3 = np.dot(npArr4, npArr3) # matrix inner product (3x3 행렬)
npResult4 = np.dot(npArr4, npArr5) # matrix inner product (3x1 벡터)

print(npResult1)
print(npResult2)
print(npResult3)
print(npResult4)

'''
 신경망에 적용
'''
preNodes = np.array([1,2]) # 신경망의 이전계층의 노드 값
weight = np.array([[1,3,5],[2,4,6]]) # 가중치
netInput = np.dot(preNodes, weight) # 노드값과 가중치의 연산
print(netInput)