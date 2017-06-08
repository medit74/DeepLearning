'''
Created on 2017. 5. 10
@author: Byoungho Kang
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
Training Set
y = ax+b (a=0.22, b=0.78)
'''
xPoint = []
yPoint = []
for idx in range(200):
    # np.random.normal(평균, 표준편차) : Draw random samples from a normal (Gaussian) distribution.
    x = np.random.normal(0.0, 0.5) 
    y = 0.22*x + 0.78 + np.random.normal(0.0, 0.1)
    xPoint.append(x)
    yPoint.append(y)
plt.plot(xPoint, yPoint, 'o', label="Input Data")
plt.legend()
plt.show()

'''
Build Model
'''
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
W = tf.Variable(tf.random_normal([1], -1.0, 1.0), name = "weight")
b = tf.Variable(tf.zeros([1]), name = "bias")
# model
model = W*X + b
# loss function
cost = tf.reduce_mean(tf.square(model - Y))
# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

'''
Train Model
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
weightList  = []
biasList    = []
costList    = []
for step in range(2001):
    costVal, weightVal, biasVal, _ = sess.run([cost, W, b, optimizer], feed_dict={X:xPoint, Y:yPoint}) 
    weightList.append(weightVal)
    biasList.append(biasVal)
    costList.append(costVal)
    if step%20 == 0:
        print(step, costVal, weightVal, biasVal)

    if step%500== 0:
        plt.plot(xPoint, yPoint, 'o', label='step={}'.format(step))
        plt.plot(xPoint, weightVal*xPoint+biasVal)
        plt.legend()
        plt.show()

plt.plot(weightList, costList)
plt.xlabel("weight")
plt.ylabel("loss")
plt.show()

