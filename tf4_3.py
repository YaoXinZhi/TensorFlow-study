#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:10:38 2019

@author: yaoxinzhi
"""

'''
滑动平均(影子值) 记录每个参数一段时间内过往值的平均 增加了模型的泛化性
使得训练在基于后期时趋于稳定的一个模型
通常针对所有参数 w b 
影子 = 衰减率 * 影子 + (1 - 衰减率) * 参数 
影子初值 = 参数初值
衰减率 = min{ MOVING_AVERAGE_DECAY, (1+轮数)/(10+轮数) }


ema = tf.train.ExponrntialMovingAverage(
        衰减率(超参数) MOVING_AVERAGE_DECAY,
        当前轮数 GLOVAL_STEP)

对括号内参数求滑动平均
ema_op = ema.apply([])
对所有参数求滑动平均
ema_op = ema.apply([tr.trainable_variables()])

计算滑动平均和训练过程 合成 一个训练结点
with tr.control_dependencies([train_step, ema_op]):
    train_op = tf.no_op(name='train')
    
返回特定参数的滑动平均值
ema.average(参数名)
'''

import tensorflow as tf

# 定义 变量 及 滑动平均类
w1 = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)

MOVING_AVERAGE_DECAY = 0.99
# 定义滑动平均
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
# 每次运行 sess.run(ema_op) 时 更新列表中元素求滑动平均值 
# 定义滑动平均结点
# ema_op = ema.apply(tf.trainable_variables())
ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
# ema.average(w1)获取 w1 滑动平均值 
# 要运行多个节点 作为列表元素列出 写在 sess.run 中
# 打印出初始参数 w1 和 滑动平均值 = w1 初值   
    print (sess.run([w1, ema.average(w1)]))
    
# 参数 w1 赋值为 1
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print (sess.run([w1, ema.average(w1)]))
    
# 更新 step 和 w1 的值 模拟出 100 轮迭代后 参数 w1 变为10
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print (sess.run([w1, ema.average(w1)]))
    
# 每次 sess.run 会更新一次w的滑动平均值  观察 影子 追随 参数  变化的过程  
    sess.run(ema_op)
    print (sess.run([w1, ema.average(w1)]))
    
    sess.run(ema_op)
    print (sess.run([w1, ema.average(w1)]))
    
    sess.run(ema_op)
    print (sess.run([w1, ema.average(w1)]))
    
    sess.run(ema_op)
    print (sess.run([w1, ema.average(w1)]))
    
    sess.run(ema_op)
    print (sess.run([w1, ema.average(w1)]))
    