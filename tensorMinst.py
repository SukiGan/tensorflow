# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:16:43 2019

@author: Cand Gan
"""

import tensorflow as tf
from tensorflow.exampls.tutorials.minst import input_data

#MISNT数据集相关的常数 
INPUT_NODE = 784  #输入的节点数 ,相当于图片的像素 
OUTPUT_NODE = 10  #输出的节点数。等于类别的数目 在MISNST中012345679 
# 配置神经网络参数

LAYER1_NODE = 500 #隐藏层的节点数 ，隐藏层有500个节点

BATCH_SIZE = 100 #一个训练的batch的训练集数据个数

LEARNING_RATE_BASE = 0.8 # 基础学习率

LEARNING_RATE_DECAY = 0.99 # 学习衰减率

REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化在损失函数中的系数

TRAINING_STEPS = 30000  #训练轮数

MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

#一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果
#定义一个使用ReLU激活函数的三层全连接神经网络，通过加入隐藏层实现多层网络结构
#ReLU实现去线性化，在这个函数中也支持传入用于计算参数平均值的类
#方便在测试时使用滑动平均模型
#

def inference(input_tensor, avg_class, weights1, biases1,
              weights2, biases2):
    #当没有使用滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        #计算隐藏层的前向传播结果，这里使用ReLU()函数
        layer1=tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        
        #计算输出层的前向传播结果。在计算损失函数时会进一步计算softmax()函数
        #这里不需要加入激活函数，不加入softmax()不会影响预测结果
        #预测结果使用的是不同类别对应节点输出值的相对大小。softmax()对于最后的分类没有影响
        #
        return tf.matmul(layer1, weights2) + biases2
    else:
        #使用avg_class.avaerge函数来计算得出变量的滑动的平均值
        #然后计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(
                tf.matmul(input_tensor, avg_class.average(weights1)) + 
                avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
        
#训练模型的过程
def train(minst):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    #生成隐藏层的参数
    weights1 = tf.Variable(
            tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    
    #生成输出层的参数
    weights2 = tf.Variable(
            tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    
    #计算当前参数下神经网络前向传播的结果，这里给出的用于计算的滑动平均的类为None
    #所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    #定义存储训练轮数的变量，这个变量不需要计算滑动平均值，
    #这里的这个变量为不可训练的变量（trainable=False）
    #在使用tensorflow神经网络训练时，一般都将将训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)
    
    #给定滑动平均衰减率和训练轮数的变量 初始化滑动平均类 
    #定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    
    #在所有代表神经网络参数的变量上使用滑动平均，
    variables_averages_op=variable_averages.aapply(
            tf.trainable_variables())
    
    #计算使用滑动平均之后的前向传播结果
    average_y = inference(
            y, variable_averages, weights1, biases1, weights2, biases2)
    
    
    #sparse_softmax_cross_entroy_with_logits计算交叉熵
    #刻画预测值和真实值之间的差距
    #
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            y, tf.argmax(y_, 1))
    #计算当前在batch中所有样例的交叉熵的平均值
    cross_entroy_mean = tf.reduce_mean(cross_entropy)
    
    #计算L2的正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    #计算模型的正规划损失，一般只计算神经网络边上权重的损失，而不适用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    
    #总损失等于交叉熵损失和正则化损失之和
    loss = cross_entroy_mean + regularization
    
    #设置指数衰减的学习率 \
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            minst.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY)
    
    
    #使用tf.train.GradientDescentOptimizer优化损失函数
    #损失包括交叉熵的损失和正则化损失
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    
    #在训练神经网络模型时，每过一遍数据需要反向传播来更新神经网络的参数
    #又要更新每一个参数的滑动平均值
    #tf.control_dependencies tf.group 两种机制
    #train_op = tf.group(train_step, variables_averages_op)等价的
    
    
with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name = 'train')
    
    
#检验使用了滑动平均模型的神经网络的前向传播结果是否正确 tf.argmax(average_y, 1)
#计算样例的每一个预测答案，averages_y = batch_size*10 的二维数组
#一行的样例就是一个前向传播结果。
#tf.argmax的第二个参数“1”表示选取最大值仅在一个维度进行
#
correct_prediction = tf.equal(tf.argmax(average_y, 1),tf.argmax(y_, 1))

#运算先将bool型转化成实数型，计算平均值
#平均值就是一组数据的正确率
accuracy = tf.reduce_mean(tf.cast(correct_perdiction, tf.float32))

#初始化会话并开始训练过程
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    
    #准备验证数据。在神经网络的训练过程中都会通过验证数据来判断停止de
    #条件和评判训练的效果
    validata_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels}
    
    #准备测试数据。
    test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels}
    
    #迭代的训练神经网络
    for i in range(TRAINING_STEPS):
        #每1000轮输出一次结果
        
        if i%1000 == 0:
            validate_acc = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d traing step(s), validation accuracy"
                  "using average model is %g" % (i, validate_acc))
        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_op, feed_dict={x: xs, y_: ys})
        
    #在训练结束之后，在测试模型上检测神经网络的最终准确率
    
    test_acc = sess.run(accuracy, feed_dict = test_feed)
    print("After %d training step, test  accuracy using average"
          "Model is %g" % (TRAINING_STEPS, test_acc))
    
#主函数
def main(argv=None):
    mnist = input_data.read_data_set("tmp/data",one_hot=True)
    train(mnist)


if __name__=='__main__':
    tf.app.run()
        
        
    
    
    
    
    
    
