#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:33:41 2019

@author: yaoxinzhi
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''
正则化缓解过拟合
在损失函数中引入模型复杂度指标 利用给 w 加权重 弱化训练的噪声 (一般不正则化b)
loss = loss(y 与 y_) + REGULARIZER*loss(w)
loss(y 与 y_) 模型中所有的参数的损失函数
REGULARIZER 用超参数给出 w 在 总loss 中的比例 及正则化的权重
loss(w) 需要正则化的参数

# 选择正则化 l1 l2
loss(w) = tf.contrib.layers.l1_regularizer(REGULARIZER)(w)
loss(w) = tf.contrib.layers.l2_regularizer(REGULARIZER)(w)

# 将正则化好的参数 加到 losses 集合
tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))

# 将 losses 集合中所有值相加 再加上交叉熵
loss = cem + tf.add_n(tf.get_collection('losses'))
'''

'''
数据 X[x0, x1] 为正态分布随机点
标注 Y_ 当 x0^2 + x1^2 < 2 时 y_ = 1  (red), 其余 y_=0 (blue)
plt.scatter(x坐标， y坐标， c="颜色")
plt.show()

# x y 轴分别打点 分别作为 x y 轴网格坐标点
xx, yy = np.mgrid[起:止:步长，起:止:步长]

把 xx yy 坐标拉值 作为矩阵 收集网格左边点
grid = np.c_[xx,ravel(), yy.ravel()] 组成矩阵

喂给神经网络训练 推测结果 y 即区域中所有坐标点颜色的量化值
probs = sess.run(y, feed_dict={c:grid})
probs = probs.reshape(xx.shape)

瞄上颜色
plt.contour(x坐标值，y坐标值，该点的高度，lebels=[等高线的高度])
plt.show()
'''

BATCH_SIZE = 30
seed = 2

rdm = np.random.RandomState(seed)
# 产生 300 个坐标 即 300*2
X = rdm.randn(300, 2)
# 标注 Y_ 当 x0^2 + x1^2 < 2 时 y_ = 1  (red), 其余 y_=0 (blue)
Y_ = [int(x0*x0 + x1*x1 < 2) for (x0, x1) in X]
# 赋值颜色 同时改为 300*1 的 label 列
Y_c = [['red' if y else 'blue'] for y in Y_]

# 对数据集 X 和 标签 Y 进行shape 整理，第一个元素为 -1 表示n行，第二个元素表示多少列
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)
#print (X)
#print (Y_)
#print (Y_c)

# np.squeeze() 从数组的形状中删除单维度条目 即把 shape 中为 1 的 维度去掉 用于画图
# 取 横 纵 坐标 颜色 并可视化
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.show()

# 定义神经网络的输入 参数 输出 定义前向传播 正则化权值
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    # 正则化的参数添加到 losses 集合
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    return b

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2,11], 0.01)
b1 = get_bias([11])
# 非线性激活函数
y1 = tf.nn.relu(tf.matmul(x, w1)+b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
# 输出层不过激活函数
y = tf.matmul(y1, w2) + b2

# 定义损失函数
# 均方误差损失函数
loss_mse = tf.reduce_mean(tf.square(y - y_))
# 均方误差 + 正则化损失 即将losses 集合中正则化的参数也加起来
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 定义反向传播方法 不含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
# 定义反向传播方法 包含正则化
#train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict= {x:X[start:end], y_:Y_[start:end]})
        if i % 2000 ==0:
            loss_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
            print ('After {0} steps, loss is: {1}'.format(i, loss_v))
# xx 在 -3 到 3 之间以步长为0.01, yy在-3到3之间以步长为0.01,生成二维网络坐标点
# xx yy 为 600*600 矩阵
    xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
# 将 xx yy 拉直 并合并成一个二列的矩阵， 得到一个网格坐标点的集合
    grid = np.c_[xx.ravel(), yy.ravel()]
# 将网格坐标点喂入神经网络 probs 为输出 即预测该点的高度(颜色)
    probs = sess.run(y, feed_dict={x:grid})
# probs 的 shape 调整为 xx 的样子
    probs = probs.reshape(xx.shape)
    print ('w1:\n{0}'.format(sess.run(w1)))
    print ('b1:\n{0}'.format(sess.run(b1)))
    print ('w2:\n{0}'.format(sess.run(w1)))
    print ('b2:\n{0}'.format(sess.run(b2)))
    
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
# 画等高线
print ('probs:\n{0}'.format(probs))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
    
# 结果 使用正则化 等高线更加平滑 数据集噪声的模型影响更小