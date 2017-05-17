'''
Created on 2017. 5. 17.
@author: Byoungho Kang
'''

import tensorflow as tf

'''
 Initialize Tensors
'''
a = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1]), dtype=tf.float32)
x = tf.placeholder(tf.float32)
print(a, b, x)

'''
 Buid a Graph
'''
y = a*x + b
print(y)

'''
 Run a Graph
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a_val, b_val, y_val = sess.run([a, b, y], feed_dict={x:1.1})
    print(a_val, b_val, y_val)
    
    