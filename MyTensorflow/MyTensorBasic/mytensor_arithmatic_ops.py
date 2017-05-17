'''
Created on 2017. 5. 17.
@author: Byoungho Kang
'''

import tensorflow as tf

'''
 Initialize Tensors
'''
x1 = tf.constant(1.0)
x2 = tf.constant(2.0)
print(x1, x2)

'''
 Buid a Graph
  - Arithmatic Operators (https://www.tensorflow.org/api_guides/python/math_ops)
'''
node1 = tf.add(x1, x2)
node2 = tf.subtract(x1, x2)
node3 = tf.multiply(x1, x2)
node4 = tf.divide(x1, x2)
print(node1, node2, node3, node4)

'''
 Run a Graph
'''
with tf.Session() as sess:
    node1_val, node2_val, node3_val, node4_val = sess.run([node1, node2, node3, node4])
    print(node1_val, node2_val, node3_val, node4_val)