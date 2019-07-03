# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:31:02 2019

@author: Cand Gan 
"""

#import tensorflow package
#import random package 

import tensorflow as tf

#定义一个变量用于计算滑动平均，初始值=0，
#手动设置变量的类型tf.float32 。所有计算滑动平均的变量必须是实数
v1 = tf.Variable(0, dtype=tf.float32)

# step 模拟神经网络迭代次数 用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均类 class 初始化时给定衰减率 0.99 和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step) 

#定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作时
#这个列表的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    #初始化所有变量
    #Use tf.global_variable_initializer ---instead 
    init_op = tf.initialize_all_variables();
    sess.run(init_op)
    
    #通过ema.average（v1）获取滑动平均之后的变量的取值，在初始化变量v1的值
    #后和v1的滑动平均都为0  
    print(sess.run([v1, ema.average(v1)]))
    
    #更新 v1=5
    sess.run(tf.assign(v1,5))
    #更新v1的滑动平均值 衰减率为 min{0.99, (1+step)/（10+step） = 0.1}
    #所以v1的滑动平均 更新为 0.1*0+0.9*5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    
    #更新step值
    #更新v1值
    sess.run(tf.assign(step,1000))
    sess.run(tf.assign(v1,10))
    
    #min{0.99, (1+step)/(10+step)}
    #0.99*4.5+0.01*10
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    
    #再次更新滑动平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    
    
#感想
#学习深度学习感觉到有点吃力了 看不懂很多东西，但代码 还得继续敲下去
# 今天参加学校的 我爱你中国 的视频短片 开心
# ，明天继续敲着代码 学习ing
    
    
    
    
    