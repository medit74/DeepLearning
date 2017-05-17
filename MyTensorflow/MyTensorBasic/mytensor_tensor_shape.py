'''
Created on 2017. 5. 17.
@author: Byoungho Kang
'''

import tensorflow as tf

'''
 Initialize Tensors
'''
m1 = tf.constant([[[1, 2, 5],
                   [3, 0, 7]],
                  [[6, 5, 9],
                   [4, 8, 3]]])
print(m1)

'''
 Buid a Graph
  - Tensor Shapes and Shaping (https://www.tensorflow.org/api_guides/python/array_ops)
'''
node1 = tf.reshape(m1, shape=(-1,3))
node2 = tf.reshape(m1, shape=(-1,1,3))
print(node1, node2)

'''
 Run a Graph
'''
with tf.Session() as sess:
    node1_val, node2_val = sess.run([node1, node2])
    print(node1_val)
    print(node2_val)