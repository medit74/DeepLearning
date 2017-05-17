'''
Created on 2017. 5. 17.
@author: Byoungho Kang
'''

import tensorflow as tf

'''
 Initialize Tensors
'''
m1 = tf.constant([[1.0,2.3,0.7],[3.9,1.0,9.3]])
print(m1)

'''
 Buid a Graph
  - Sequence Comparison and Indexing (https://www.tensorflow.org/api_guides/python/math_ops)
'''
node1 = tf.argmax(m1, axis=0)
node2 = tf.argmax(m1, axis=1)
node3 = tf.argmax(m1, axis=-1)
print(node1, node2, node3)

'''
 Run a Graph
'''
with tf.Session() as sess:
    node1_val, node2_val, node3_val = sess.run([node1, node2, node3])
    print(node1_val, node2_val, node3_val)