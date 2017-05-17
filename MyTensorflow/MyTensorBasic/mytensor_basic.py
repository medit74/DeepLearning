'''
Created on 2017. 4. 17.

@author: Byoungho Kang
'''

import numpy as np
import tensorflow as tf

m1 = tf.constant([[1.,2.],[3.,4.]])
m2 = tf.constant([[1.],[2.]])
m12 = tf.matmul(m1, m2)

sess = tf.Session()
print(sess.run(m1))
print(sess.run(m2))
print(sess.run(m12))
print(sess.run(m1*m2))
print(sess.run(m1+m2))

print("--- axis 연산 ---")
print(sess.run(tf.reduce_mean(m1)))
print(sess.run(tf.reduce_mean(m1, axis=0)))
print(sess.run(tf.reduce_mean(m1, axis=1)))
print(sess.run(tf.reduce_mean(m1, axis=-1)))

a = tf.constant([[1.0,2.3,0.7],[3.9,1.0,9.3]])
print(sess.run(tf.argmax(a, axis=0)))
print(sess.run(tf.argmax(a, axis=1)))

print("--- reshape 연산 ---")
c = np.array([[[1.,2.,5.],
               [3.,0.,7.]],
              [[6.,5.,9.],
               [4.,8.,3.]]])
print(c, c.shape)
tf.reshape(c, shape=(-1,3))
print(tf.shape(c))
print(sess.run(tf.reshape(c, shape=(-1,3))))
print(sess.run(tf.reshape(c, shape=(-1,1,3))))


print("-- placeholder & variables --")
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
Y = a+b
print(sess.run(Y, feed_dict={a:8, b:2.3}))
print(sess.run(Y, feed_dict={a:[2,3], b:[5,9]}))

a = tf.Variable(tf.random_normal([784, 200]), name="Var")
b = tf.Variable(tf.zeros([200]))
print(a, b)
