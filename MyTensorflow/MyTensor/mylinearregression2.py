'''
Created on 2017. 4. 17.

@author: Byoungho Kang
'''

import tensorflow as tf
import matplotlib.pyplot as plt

'''
Build Model
y = X*W+b (tf.matmul, np.dot)
X.shape (3x5)
W.shppe (5x1)
b.shape (1)
Y.shape (3x1)
'''
inputData = [[23., 35., 48., 62., 83.]
            ,[38., 46., 59., 74., 95.]
            ,[35., 42., 46., 53., 60.]]
label = [[110.], [198.], [257.]]

#X = [1,2,3]
#Y = [7,12,17]
X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([5,1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

model = tf.matmul(X,W) + b

# loss function
cost = tf.reduce_mean(tf.square(model - Y))
# Gradient Descent (Gradient optimization)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)

'''
Initialize Model
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

'''
Train Model
'''
costList = []
weightList = []
biasList = []
modelList = []

for step in range(3001):
    costVal, weightVal, biasVal, modelVal, _ = sess.run([cost, W, b, model, optimizer], feed_dict={X:inputData, Y:label})
    costList.append(costVal)
    weightList.append(weightVal)
    biasList.append(biasVal)
    modelList.append(modelVal)
    if step%20 == 0:
        print(step, "\ncostVal", costVal, "\nweightVal", weightVal, "\nbiasVal", biasVal, "\nmodelVal", modelVal)

plt.plot(costList)
plt.xlabel("step")
plt.ylabel("cost")
plt.show()
