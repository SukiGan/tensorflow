# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:11:30 2019

@author: Cand Gan
"""
#深度学习框架
import tensorflow as tf
#导入科学计数包，生成模拟数据
from numpy.random import RandomState

#定义一个模拟的数据集
batch_size = 8

#定义神经网络参数
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None,2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1), name='y-input')

#定义神经网络的传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)

#随机生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
#定义规则来给出样本的标签
Y = [[int(x1+x2 <1)] for (x1,x2) in X]

#创建会话来运行一个tensorflow程序 
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    #初始化变量
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    
    # 设置训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+ batch_size, dataset_size)
    
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        
        if (i % 1000 ==0):
            total_cross_entropy = sess.run(
                    cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d  train steps, cross_entropy on all data is %g"
                  %(i, total_cross_entropy))
        
        print(sess.run(w1))
        print(sess.run(w2))
