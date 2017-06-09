'''
Created on 2017. 5. 18.
@author: Byoungho Kang
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Training Set
The MNIST data is split into three parts: 
 - 55,000 data points of training data (mnist.train), 
 - 10,000 points of test data (mnist.test), and 
 - 5,000 points of validation data (mnist.validation)
'''
mnist = input_data.read_data_sets("../resources/", one_hot=True)
'''
for idx in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs_val, batch_ys_val = mnist.validation.next_batch(100)
    print(idx, batch_xs.shape, batch_ys.shape, batch_xs_val.shape, batch_ys_val.shape)
'''    

'''
Hyper Parameter
'''
tf.flags.DEFINE_float("lr", 0.1, "learning rate")
batch_size = 100
#learning_rate   = 0.01
'''
Build a Model
'''
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.random_normal([10]))

logits = tf.matmul(X, W) + b 
predictions = tf.nn.softmax(logits)
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=predictions)

# accuracy
correct_prediction  = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
accuracy            = tf.reduce_mean(tf.cast(correct_prediction, "float"))

tf.summary.histogram("W", W)
tf.summary.histogram("b", b)
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merge_op = tf.summary.merge_all()

optimizer = tf.train.GradientDescentOptimizer(learning_rate=tf.flags.FLAGS.lr).minimize(loss)

'''
Train Model
'''
with tf.Session() as sess:
    writer_train = tf.summary.FileWriter('../logs/train', sess.graph)
    writer_valid = tf.summary.FileWriter('../logs/valid', sess.graph)
    
    sess.run(tf.global_variables_initializer())
    for step in range(10000):
        batch_images, batch_labels = mnist.train.next_batch(batch_size)
        feed = {X: batch_images, Y: batch_labels}
        loss_val, accuary_val, summary, _ = sess.run([loss, accuracy, merge_op, optimizer], feed_dict=feed)
        if step % 10 == 0 :
            print ("step {} | loss {:.9f}".format(step, loss_val))
            writer_train.add_summary(summary, step)
        
        batch_images_valid, batch_labels_valid = mnist.validation.next_batch(batch_size)
        feed = {X: batch_images_valid, Y: batch_labels_valid}
        loss_val, accuary_val, summary, _ = sess.run([loss, accuracy, merge_op, optimizer], feed_dict=feed)
        if step % 10 == 0 :
            print ("step {} | loss {:.9f}".format(step, loss_val))
            writer_valid.add_summary(summary, step)
        
#print(tf.flags.FLAGS.lr)