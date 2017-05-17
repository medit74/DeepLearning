'''
Created on 2017. 5. 17.
@author: Byoungho Kang
'''

import tensorflow as tf

'''
 Tensor : The central unit of data in Tensorflow
  - rank : number of dimentions (1-D Tensor, 2-D Tensor...) 
'''
# Constant 0-D Tensor (Scala)
tensor0_a = tf.constant(0, dtype=tf.int16, name="tensor0_a")
tensor0_b = tf.constant(0.1, dtype=tf.float64, name="tensor0_b")
print(tensor0_a, tensor0_b)

# Constant 1-D Tensor (Vector)
tensor1_a = tf.constant([1,2,3,4,5])
tensor1_b = tf.constant(1.1, shape=[5], dtype=tf.float16)
print(tensor1_a, tensor1_b)

# Constant 2-D Tensor (Matrix)
tensor2_a = tf.constant([[1,2,3], [4,5,6]])
tensor2_b = tf.constant(2.1, shape=[3,4])
print(tensor2_a, tensor2_b)

# Constant 3-D Tensor
tensor3_a = tf.constant([[[1],[2],[3]], [[4],[5],[6]], [[7],[8],[9]]])
tensor3_b = tf.constant(True, shape=[3,4,5])
print(tensor3_a, tensor3_b)

# Constant 4-D Tensor
tensor4_a = tf.constant([[[[1,2,3,4],[5,6,7,8],[9,10,11,12]], [[13,14,15,16],[17,18,19,20],[21,22,23,24]]]])
tensor4_b = tf.constant('hello', shape=[2,3,4,5])
print(tensor4_a, tensor4_b)

# Random 3-D Tensor & Zero 1-D Tensor, Ones 1-D Tensor
tensor_norm = tf.random_normal(shape=[3,4,5], mean=-1, stddev=4)
tensor_zero = tf.zeros(shape=[5])
tensor_ones = tf.ones(shape=[5])
print(tensor_norm, tensor_zero, tensor_ones)
