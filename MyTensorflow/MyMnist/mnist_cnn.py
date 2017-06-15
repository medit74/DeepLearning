'''
Created on 2017. 5. 10
@author: Byoungho Kang
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

'''
Functions for Weight and Bias
 - truncated normal distribution - 정규분포에서 일부 구간을 잘라낸 분포
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.zeros(shape=shape)
    return tf.Variable(initial)

'''
Training Set
'''
mnist = input_data.read_data_sets("../resources/", one_hot=True)

'''
Hyper Parameter
'''
training_epochs = 5
batch_size      = 100
learning_rate   = 0.001
display_step    = 10

'''
Build Model
    -----------------------------------------------------------------------------------
    input size | 전체 사이즈를 28*28*1로 처리
    -----------------------------------------------------------------------------------
    conv1      | filter 5*5*1,  32ea >> n*28*28*32
               | pool 2*2            >> n*14*14*32
    conv2      | filter 5*5*32, 64ea >> n*14*14*64
               | pool 2*2            >> n*7*7*64
    fc1        | reshape             >> 7*7*64, m   >> for fully-connected
    out        | classes number      >> m, 10
    -----------------------------------------------------------------------------------
'''
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
dropout_rate = tf.placeholder(tf.float32)

# First Convolution Layer (28x28x1 -> 28x28x32 -> 14x14x32)
X_image = tf.reshape(X, [-1, 28, 28, 1]) # 4D Tensor, -1은 나누어 떨어지는 숫자는 자동으로 계산해 적용하겠다라는 의미
W_conv1 = weight_variable([5, 5, 1, 32]) # 필터 5x5x1 크기 32개
b_conv1 = bias_variable([32]) # 필터별 하나의 Bias

h_conv1 = tf.nn.relu(tf.nn.conv2d(X_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Second Convolution Layer (14x14x32 -> 14x14x64 -> 7x7x64)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# First Fully Connected Layer (7x7x64 -> 1024)
W_fc1 = weight_variable([7*7*64, 1024]) #7*7*64개 입력, 1024 출력
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, dropout_rate)

# Second Fully Connected Layer (1024 -> 10)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# loss function : Computes softmax cross entropy between logits and labels.
crossentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = Y))
# gradient descent
optimizer    = tf.train.AdamOptimizer(learning_rate).minimize(crossentropy)
# accuracy
correct_prediction  = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
accuracy            = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) ## cast: same shape as x

'''
Train Model
'''
epochList = []
costList  = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
            
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            cost, _ = sess.run([crossentropy, optimizer], feed_dict = {X:batch_xs, Y:batch_ys, dropout_rate:0.7})
            avg_cost += cost/total_batch
            if i % display_step == 0 :
                cost, acc = sess.run([crossentropy, accuracy], feed_dict = {X:batch_xs, Y:batch_ys, dropout_rate:1.})
                print("Iter " + str(i) + ": Minibatch Cost= " + "{:.6f}".format(cost) + ", Accuracy= " + "{:.5f}".format(acc))
            
        print("Epoch:","%03d" % (epoch+1), ", Avg.Cost: {:.9f}".format(avg_cost))
        epochList.append(epoch+1)
        costList.append(avg_cost)
    
    plt.plot(epochList, costList, 'o', label="MNIST Convolution Neural Network")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()

    ''' Testing ''' 
    print("Accuracy: ", sess.run(accuracy, feed_dict = {X: mnist.test.images[:1000], Y:mnist.test.labels[:1000], dropout_rate:1.} ))
    
    ''' Random Predict ''' 
    r = random.SystemRandom().randint(0, mnist.test.num_examples -1)          
    print(r)
    print("Label: ",   sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))) 
    print("Predict: ", sess.run(tf.argmax(y,1), {X:mnist.test.images[r:r+1], dropout_rate:1.}))
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    plt.show()    
