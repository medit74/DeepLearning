'''
Created on 2017. 4. 17.
@author: Byoungho Kang
'''

import numpy as np
import tensorflow as tf

'''
Training Set
 data https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data
 X : 16 digit, Y : 1 digit (1~7의 분류 번호)
'''
zooData = np.loadtxt('../resources/zoo.data.txt', dtype="int32", delimiter=',')
x_data = zooData[:, 0:-1]  # data
y_data = zooData[:, [-1]]  # label
print(zooData, zooData.ndim, zooData.shape, zooData.size, zooData.dtype)
print(x_data.ndim, x_data.shape, x_data.size, x_data.dtype)
print(y_data.ndim, y_data.shape, y_data.size, y_data.dtype)

# Y의 값을 one-hot encoding으로 변경
temp = np.zeros((y_data.size, 7))
print(temp.ndim, temp.shape, temp.size, temp.dtype)
for idx, row in enumerate(temp):
    row[y_data[idx]-1] = 1
    print(idx, row, y_data[idx])
y_data = temp
print(y_data.ndim, y_data.shape, y_data.size, y_data.dtype)

'''
Build Model 
 y = X*W+b 
 X.shape (batch, 16)
 W.shppe (16x7)
 b.shape (7)
 y.shape (batch, 7) 
'''
X = tf.placeholder(shape = [None, 16], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 7],  dtype = tf.int32); 
W = tf.Variable(tf.random_normal([16, 7]), name = "weight")
b = tf.Variable(tf.random_normal([7]), name = "bias")

## model
y = tf.matmul(X, W) + b
## loss 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = Y))
## optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-4).minimize(cost)
## accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
Train a Model
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5000):
        costVal, _, accuracyVal = sess.run([cost, optimizer, accuracy], feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, costVal, accuracyVal))
                
    ''' Predict '''
    for idx in range(x_data.shape[0]):
        predict = sess.run(tf.argmax(y, 1)+1, feed_dict={X:x_data[idx].reshape(1,16)}) 
        print("Step: {:3}".format(idx), predict, zooData[idx, [-1]])
