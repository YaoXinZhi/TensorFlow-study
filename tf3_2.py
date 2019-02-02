# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 21:28:00 2019

@author: yaoxinzhi
"""

'''
该代码用于跟着 北大 人工智能实践: TensorFlow 笔记 课程代码训练
'''

import tensorflow as tf
import numpy as np

#==============================================================================
# # 仅仅构建了运算图 
# a = tf.constant([1.0, 2.0])
# b = tf.constant([3.0, 4.0])
# result = a + b
# print (result)
#==============================================================================

#==============================================================================
# # x一行两列 y两行一列
# x = tf.constant([[1.0, 2.0]])
# w = tf.constant([[3.0], [4.0]])
# # y = xw = x1*w1 + x2*w2
# y = tf.matmul(x, w)
# print (y)
# 
# # Session 即计算
# with tf.Session() as sess:
#         print (sess.run(y))
#==============================================================================

'''
tf 参数 及 权重
正态分布 产生2*3矩阵 标准差为2 平均值为0 随机种子
 w = tf.Variable(tf.random_normal([2,3], stddev=2, mean=0, seed=1))
tf.truncated_normal() 去掉过大偏离点的正态分布
tf.random_uniform() 平均分布
tf.zeros() tf.ones() 
tf.fill() 权定值数组 tf.fill([3,2], 6) 生成 [[6,6], [6,6], [6,6]]
tf.constant() 直接给值 tf.constan([3, 2, 1]) 生成 [3, 2, 1]
'''

'''
神经网络实现：
准备数据集 提取特征 喂给神经网络
搭建NN结构 正向传播
大量特征数据喂给NN 反向传播 优化NN参数
使用训练好的模型预测和分类
'''

'''
前向传播 搭建模型 实现推理
x 输入层一行两列 一次输入两个特征 体积 重量
w1 前面2结点 后面3结点 2*3
a1 为第一个计算层（第一层网络） 1*3 = x*w1
w2 3*1
y = a1*w2
两层全连接
'''

#==============================================================================
# # 定义输入和参数
# x = tf.constant([[0.7, 0.5]])
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
# 
# # 定义前向传播过程
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
# 
# # 用会话计算结果
# with tf.Session() as sess:
# # 变量初始化
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     print ('y in tf3_3.py is: \n {0}'.format(sess.run(y)))
#==============================================================================

    
# 如果输入不是初始化 而是喂的用tf.placeholder 占位， 在sess.run 函数中用 feed_dict 喂数据
#==============================================================================
# # 喂一组数据 2个输入特征
# #x = tf.placeholder(tf.float32, shape=(1, 2))
# #  喂多组数据
# x = tf.placeholder(tf.float32, shape=(None, 2))
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
#  
#  # 定义前向传播过程
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
#  
#  # 用会话计算结果
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
# # 喂一组数据 2个输入特征
# #    print ('y2 is: \n {0}'.format(sess.run(y, feed_dict={x: [[0.7, 0.5]]})))
# #  喂多组数据
#     print ('y3 is : \n:{0}'.format(sess.run(y, feed_dict={x: [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]})))
#==============================================================================

'''
反向传播 训练模型参数 梯度下降
使模型在训练数据上的损失函数最小
即 最小化 预测值 y 与已知答案 y_ 的差距

常用 均方误差
loss = tf.reduce_mean(rf.square(y_-y))
以减小loss值为目标

优化器：
learning_rate 参数每次更新的幅度
train_step = tf.train.GradientDescentOptimiGradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
''' 

# BARCH_SIZE 每次喂入模型数据
BATCH_SIZE = 8
# 随机种子值
seed = 23455

# 基于seed产生随机数
rng = np.random.RandomState(seed)
# 随机数返回32行2列的举证 32组数据 体积和重量 作为输入数据集 32*2
X = rng.rand(32, 2)
# 从X这个32行2列的举证中 取出一行 判断如果和小于 1 给 Y 赋值 1 如果和不小于 1 给 Y 赋值 0
# 作为数据数据集的标签（正确答案） 32*1
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print ("X:\n{0}".format(X))
print ("Y:\n{0}".format(Y))

# 定义神经网络的输入 参数 和 输出 定义前向传播过程 None*2
x = tf.placeholder(tf.float32, shape=(None, 2))
# 正确答案 32*1
y_ = tf.placeholder(tf.float32, shape=(None,1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数 及 反向传播方法
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

 #生成会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前 未经训练 的 参数取值
    print ('w1: \n{0}'.format(sess.run(w1)))
    print ('w2: \n{0}'.format(sess.run(w2)))
    print ('\n')
    
    # 训练模型
    STEPS = 5000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 ==0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print ("After {0} training step(s), loss on all data is {1}".format(i, total_loss))
    # 输出训练后的参数取值
    print ('\n')
    print ('w1:\n{0}'.format(sess.run(w1)))
    print ('w2:\n{0}'.format(sess.run(w2)))