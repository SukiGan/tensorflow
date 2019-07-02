# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:52:54 2019

@author: Cand  Gan 
"""

import tensorflow as tf 
from numpy.random import RandomState

batch_size = 8

#两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
#回归问题 只有一个节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
#定义一个单层的神经网络前向传播。简单加和
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y =tf.matmul(x, w1)

#定义预测成本
loss_less = 10
loss_more = 1

loss = tf.reduce_sum(
        tf.where(tf.greater(y, y_),
                 (y - y_)*loss_more,
                 (y_ - y)*loss_less))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#产生一个模拟数据集 
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]

#训练神经网络 
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        print(sess.run(w1))
