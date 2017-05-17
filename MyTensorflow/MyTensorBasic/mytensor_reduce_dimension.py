'''
Created on 2017. 5. 17.
@author: Byoungho Kang
'''

import tensorflow as tf

'''
 Initialize Tensors
'''
m1 = tf.constant([[1.0,2.0],[3.0,4.0]])
print(m1)

'''
 Buid a Graph
  - Reduce various dimensions (https://www.tensorflow.org/api_guides/python/math_ops)
'''
node1 = tf.reduce_mean(m1)
node2 = tf.reduce_mean(m1, axis=0)
node3 = tf.reduce_mean(m1, axis=1)
node4 = tf.reduce_mean(m1, axis=-1)
node5 = tf.reduce_sum(m1)
node6 = tf.reduce_sum(m1, axis=-1)
print(node1, node2, node3, node4, node5, node6)

'''
 Run a Graph
'''
with tf.Session() as sess:
    node1_val, node2_val, node3_val, node4_val, node5_val, node6_val = sess.run([node1, node2, node3, node4, node5, node6])
    print(node1_val, node2_val, node3_val, node4_val, node5_val, node6_val)