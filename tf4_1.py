#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:13:07 2019

@author: yaoxinzhi
"""
import tensorflow as tf
import numpy as np
'''
增加 偏置 和 激活函数

激活函数： 避免纯线性函数
tf.nn.relu()
tf.nn.sigmoid()
tf.nn.tanh()

复杂度 
NN层数 + NN参数
层数 = 隐藏层 + 1个输出层

优化 
损失函数 预测值 y 与 已知答案 y_ 的差距
loss 最小 
loss_mse = tf.reduce_mean(tf.square(y-y_))
'''
'''
预测酸奶日销量y x1 x2 是影响日销量因素
y_ = x1 + x2 噪声: -0.05 ~ +0.05 拟合可以预测销量的函数
'''
# 定义 BATCH_SIZE SEED
BATCH_SIZE = 8
SEED = 23455

# 训练集
rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
# 定义的训练集 y =  x1 + x1 即 w1 为 [[1], [1]]
Y_ = [[x1 + x2 + (rdm.rand()/10.0-0.05)] for (x1, x2) in X]

# 定义输入 参数 输出 前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x, w1) 

# 定义损失函数 及 反向传播
# 定义损失函数为MSE
#loss = tf.reduce_mean(tf.square(y_ - y))

'''
自定义损失函数
如预测多了 损失成本 预测少了 损失利润
若 利润 != 成本 则 mse 产生的 loss 无法利益最大化

定义损失为分段函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), COST(y-y_, PROFIT(y_-y))))
定义 酸奶成本 COST 1元 酸奶利润 PROFIT 9元
预测少了 损失利润9元 / 预测多了 损失成本一元
所以预测少了损失大 希望生成的损失函数往多了预测
'''
COST = 1
PROFIT = 9
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*COST, (y_-y)*PROFIT))

'''
交叉熵 表征两个概率分布之间的距离
即哪个预测结果与标准答案更接近
loss_ce = tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-12, 1.0)))
tf.clip_by_value 对log值做限制 保证log值有意义 符合概率分布

当n分类的n个输出 通过 softmax()函数 便满足了概率分布要求  都在0,1 之间 和为1
可以直接求交叉熵
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cem = tf.reduce_mean(ce)
'''

#反向传播方法为梯度下降
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 初始化参数 喂训练集
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x:X, y_:Y_})
            print ('After {0} training steps. w1 is:\n{1}'.format(i, sess.run(w1)))
#            print ('total_loss is:\n{0}\n'.format(total_loss))
    print('Final w1 is:\n{0}'.format(sess.run(w1)))