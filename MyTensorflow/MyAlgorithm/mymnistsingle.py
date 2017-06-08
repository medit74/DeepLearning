'''
Created on 2017. 5. 10
@author: Byoungho Kang
'''

import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Training Set
The MNIST data is split into three parts: 
 - 55,000 data points of training data (mnist.train)
 - 10,000 points of test data (mnist.test)
 - 5,000 points of validation data (mnist.validation)
'''
mnist = input_data.read_data_sets("../resources/", one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(100)
print(batch_xs.shape, batch_ys.shape)

'''
Hyper Parameter
'''
training_epochs = 25
batch_size      = 100
learning_rate   = 0.001
#tf.flags.DEFINE_float("learning_rate", 0.1, "learning rate")

'''
Build a Model
'''
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

affine              = tf.matmul(X,W) + b
y                   = tf.nn.softmax(affine)
# loss function
#crossentropy        = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(y), reduction_indices=[1]))
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=y)

# gradient descent
optimizer           = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
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
            cost, _ = sess.run([loss, optimizer], feed_dict = {X:batch_xs, Y:batch_ys})
            avg_cost += cost/total_batch
            
        print("Epoch:","%03d" % (epoch+1), ", Avg.Cost: {:.9f}".format(avg_cost))
        epochList.append(epoch+1)
        costList.append(avg_cost)
    
    plt.plot(epochList, costList, 'o', label="MNIST Single Layer Perceptron")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()

    ''' Testing ''' 
    print("Accuracy: ", accuracy.eval({X: mnist.test.images, Y:mnist.test.labels}))
    print("Accuracy: ", sess.run(accuracy, feed_dict = {X: mnist.test.images, Y:mnist.test.labels} ))
    
    ''' Random Predict ''' 
    r = random.SystemRandom().randint(0, mnist.test.num_examples -1)          
    print(r)
    print("Label: ",   sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Predict: ", sess.run(tf.argmax(y,1), {X:mnist.test.images[r:r+1]}))
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    plt.show()
    