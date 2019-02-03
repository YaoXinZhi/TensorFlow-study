#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:50:23 2019

@author: yaoxinzhi
"""

'''
学习率 每次参数更新的幅度
Wn+1 = Wn - learning_rate'
'''
# 设损失函数 loos = (w+1)^2 令w初值为5 反向传播就是求最优w 即最小化loss对应的w值
import tensorflow as tf

# 定义代优化参数w初值赋5
w = tf.Variable(tf.constant(5, dtype=tf.float32))

# 定义损失函数
loss = tf.square(w+1)

# 定义反向传播方法
# 学习率太大震荡不收敛 学习率小了收敛速度1慢
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

'''
指数衰减学习率 根据 轮数 动态更新学习率
learning_rate = LEARNING_RATE_BASE * LEARNING_RATE_DECAY^(global_step / LEARNING_RATE_STEP)
LEARNING_RATE_BASE 学习率初始值
LEARNING_RATE_DECAY 学习率衰减率 （0,1）
global_step 运行了几轮 BATCH_SIZE
LEARING_RATE_STEP 每多少轮更新一次学习率 = 总样本数 / BETCH_SIZE

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        LEARNING_RATE_STEP,
        LEARNING_RATE_DECAY,
        staircase = TRUE)
staircase 多少轮更新一次取整数 学习率程梯形衰减
'''
# 初始学习率
LEARNING_RATE_BASE = 0.1
# 学习率衰减率
LEARNING_RATE_DECAY = 0.99
# 喂入多少轮BATCH_SIZE后 学习率更新一次 一般设为 总样本数/BATCH_SIZE
LEARNING_RATE_STEP = 1
# 运行几轮 BATCH_SIZE 的计数器 初值给 0 设置为不被训练
global_step = tf.Variable(0, trainable=False)
# 定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                           LEARNING_RATE_STEP, LEARNING_RATE_DECAY,
                                           staircase=True)

# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 生成会话 训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40
    for i in range(STEPS):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        global_step_val = sess.run(global_step)
        learning_rate_val = sess.run(learning_rate)
#        print ('After {0} steps: w is {1}, loss is {2}'.format(i, w_val, loss_val))
        print ('After {0} steps: global_step is {1}, w is {2}, learning rate is {3}, loss is {4}'.format(i, global_step_val, w_val, learning_rate_val, loss_val))