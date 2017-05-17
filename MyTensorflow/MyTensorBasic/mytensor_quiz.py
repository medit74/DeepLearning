'''
Created on 2017. 5. 17.
@author: Byoungho Kang
 - (3,4) matrix, (4,6) matrix를 feed 받아 matrix multiply 하는 model 생성하기 
 - feed 값은 Gaussian Distribution에서 Random 추출하기 (numpy.random.randn(d0, d1, ..., dn))
'''

import numpy as np
import tensorflow as tf

'''
 Initialize Tensors
'''
x1 = tf.placeholder(tf.float32, shape=[3,4])
x2 = tf.placeholder(tf.float32, shape=[4,6])
print(x1, x2)

'''
 Buid a Graph
'''
node = tf.matmul(x1, x2)
print(node)

'''
 Run a Graph
'''
with tf.Session() as sess:
    x1, x2, node_val = sess.run([x1, x2, node], feed_dict={x1:np.random.randn(3,4), x2:np.random.randn(4,6)})
    print(x1)
    print(x2)
    print(node_val)