'''
Created on 2017. 5. 10
@author: Byoungho Kang
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

'''
Xavier 초기값 for Weight
'''
def xavier_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev = stddev)

'''
Training Set
'''
mnist = input_data.read_data_sets("../resource/", one_hot=True)

'''
Hyper Parameter
'''
training_epochs = 25
batch_size      = 100
learning_rate   = 0.001

'''
Build Model
'''
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])
dropout_rate = tf.placeholder("float")

W1 = tf.get_variable("W1", shape = [784, 256], initializer = xavier_init(784, 256))
W2 = tf.get_variable("W2", shape = [256, 256], initializer = xavier_init(256, 256))
W3 = tf.get_variable("W3", shape = [256,  85], initializer = xavier_init(256,  85))
W4 = tf.get_variable("W4", shape = [85,   28], initializer = xavier_init(85,   28))
W5 = tf.get_variable("W5", shape = [28,   10], initializer = xavier_init(28,   10))
b1 = tf.Variable(tf.zeros([256]))
b2 = tf.Variable(tf.zeros([256]))
b3 = tf.Variable(tf.zeros([85]))
b4 = tf.Variable(tf.zeros([28]))
b5 = tf.Variable(tf.zeros([10]))

# model
_L1 = tf.nn.relu(tf.add(tf.matmul(X,  W1), b1))
L1  = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2  = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
L3  = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))
L4  = tf.nn.dropout(_L4, dropout_rate)
y = tf.add(tf.matmul(L4, W5), b5) # softmax는 아래 softmax_cross_entropy_with_logits로 적용

# loss function : Computes softmax cross entropy between logits and labels.
crossentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = Y))
# gradient descent
optimizer    = tf.train.AdamOptimizer(learning_rate).minimize(crossentropy)
# accuracy
correct_prediction  = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
accuracy            = tf.reduce_mean(tf.cast(correct_prediction, "float")) ## cast: same shape as x

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
            
        print("Epoch:","%03d" % (epoch+1), ", Avg.Cost: {:.9f}".format(avg_cost))
        epochList.append(epoch+1)
        costList.append(avg_cost)
    
    plt.plot(epochList, costList, 'o', label="MNIST Multi Layer Perceptron")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()

    ''' Testing ''' 
    print("Accuracy: ", accuracy.eval({X: mnist.test.images, Y:mnist.test.labels, dropout_rate:1.}))
    print("Accuracy: ", sess.run(accuracy, feed_dict = {X: mnist.test.images, Y:mnist.test.labels, dropout_rate:1.} ))
    
    ''' Random Predict ''' 
    r = random.SystemRandom().randint(0, mnist.test.num_examples -1)          
    print(r)
    print("Label: ",   sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))) 
    print("Predict: ", sess.run(tf.argmax(y,1), {X:mnist.test.images[r:r+1], dropout_rate:1.}))
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    plt.show()
    