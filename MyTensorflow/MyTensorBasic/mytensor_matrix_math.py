'''
Created on 2017. 5. 17.
@author: Byoungho Kang
'''

import tensorflow as tf

'''
 Initialize Tensors
'''
m1 = tf.constant([[1.0,2.0],[3.0,4.0]])
m2 = tf.constant([[5.0],[6.0]])
print(m1, m2)

'''
 Buid a Graph
  - Matrix Math Functions (https://www.tensorflow.org/api_guides/python/math_ops)
'''
node1 = tf.matmul(m1, m2)
node2 = tf.transpose(m1)
node3 = tf.matrix_inverse(m1)
print(node1, node2, node3)

'''
 Run a Graph
'''
with tf.Session() as sess:
    node1_val, node2_val, node3_val = sess.run([node1, node2, node3])
    print(node1_val)
    print(node2_val)
    print(node3_val)