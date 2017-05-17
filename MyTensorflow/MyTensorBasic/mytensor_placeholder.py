'''
Created on 2017. 5. 17.
@author: Byoungho Kang
'''

import tensorflow as tf

'''
 Initialize Tensors
'''
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
print(x1, x2)
y1 = tf.placeholder(tf.float32, shape=[2,3])
y2 = tf.placeholder(tf.float32, shape=[3])
print(y1, y2)


'''
 Buid a Graph
  - Arithmatic Operators (https://www.tensorflow.org/api_guides/python/math_ops)
'''
node1 = tf.add(x1, x2)
node2 = tf.add(y1, y2)
print(node1, node2)

'''
 Run a Graph
'''
with tf.Session() as sess:
    node1_val = sess.run(node1, feed_dict={x1:8, x2:2.3})
    print(node1_val)
    node1_val = sess.run(node1, feed_dict={x1:[1,2], x2:[3,4]})
    print(node1_val)
    
    feed = {y1:[[1,2,3],[4,5,6]], y2:[1,0,-1]}
    node2_val = sess.run(node2, feed_dict=feed)
    print(node2_val)
    
    