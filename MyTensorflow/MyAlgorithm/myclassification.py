'''
Created on 2017. 4. 17.

@author: Byoungho Kang
'''

import numpy as np
import tensorflow as tf

'''
data https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data
'''

# Seeding (?)
tf.set_random_seed(6100)

# data loading
data = np.loadtxt('zoo.data.txt', delimiter = ',')
print(data)

## Train Data Split (Classification Number 정의 y_data >> 1 ~ 7
x_data = data[:, 0:-1]   
y_data = data[:, [-1]]  # label
#print(x_data, y_data)
print(x_data.shape, y_data.shape)

## Classification Number 정의 >> 1 ~ 7
class_num = 7 

'''
Build Model 
분류갯수 7
y = X*W+b (tf.matmul, np.dot)
X.shape (1x16)
W.shppe (16x7)
b.shape (7)
Y.shape (1x7) : one-hot encoding
Y.shape (1x1) : reshape
'''
X = tf.placeholder(shape = [None, 16], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 1],  dtype = tf.int32); 
print(Y) # 1~7 사이의 값
Y_one_hot = tf.one_hot(Y, class_num)                   
print(Y_one_hot) # one_hot encoding값이 (1,7) shape
Y_one_hot = tf.reshape(Y_one_hot, [-1, class_num])     
print(Y_one_hot) # one_hot encoding값을 (7,) shape로 변환


## W, b 정의
W = tf.Variable(tf.random_normal([16, class_num]))
b = tf.Variable(tf.random_normal([class_num]))

## model 정의
model      = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(model) # 확률값으로 변환 후에 one_hot 값과 비교하여 오차계산

## loss function 정의(cost 최소화) >> softmax
##-- **_with_logits: Classification 처리 없는 상태에서 계산 (softmax 전단계)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model, labels = Y_one_hot))

## Gradient Descent (Gradient optimization)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5).minimize(cost)

## Classification Accuracy
prediction         = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ## training
    for step in range(5000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        
        ## training log
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
                
    # predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for pred_val, y_val in zip(pred, y_data.flatten()):
        print("Pred: {} | Prediction: {} | True Y: {} | [{}]".format(y_val, pred_val, int(y_val), pred_val == int(y_val)))