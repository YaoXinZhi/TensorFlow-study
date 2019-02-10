#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:35:37 2019

@author: yaoxinzhi
"""

'''
这篇注释 废话较多 但 舍不得删 故 保留
'''

import tensorflow as tf
# 图片分辨率
IMAGE_SIZE = 28
# 通道数
NUM_CHANNELS = 1
# 卷积核大小 及 数目
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
# 全连接网络NODE数目
FC_SIZE = 512
OUTPUT_NODE = 10
# dropout 舍弃概率
KEEP_PROB = 0.5

'''
Lenet-5
# 图片深度 = 卷积核个数 池化操作不改变图片深度
conv1 = tf.nn.conv2d([BATCH_SIZE, 32, 32, 1],
             [5, 5, 1, 6],
             [1, 1, 1, 1],
             padding = 'VALID')
# 输出边长 (32-5+1)/1 = 28
pool1 = tf.nn.pool([BATCH_SIZE, 28, 28, 6],
                  [2, 2, 6, 1],
                  [1, 2, 2, 1],
                  padding='SAME')
# 输出边长 28/2 = 14
conv2 = tf.nn.conv2d([BATCH_SIZE, 14, 14, 6],
                     [5, 5, 6, 16],
                     [1, 1, 1, 1],
                     padding='VALID')
# 输出步长 (14-5+1)/1 = 10
pool2 = tf.nn.pool([BATCH_SIZE, 10, 10, 16],
                   [5, 5, 16, 1],
                   [1, 2, 2, 1],
                   padding='VALID')
# 输出边长 10/2=5

# 需要注意图片深度的变化 等于上一卷积层的卷积核个数
conv2 = tf.nn.conv2d([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS],
                     [conv_size, conv_size, img_CHANNELS, conv_KERNEL_NUM],
                     [1, length_step, length_step, 1],
                     padding='SAME')

pool2 = tf.nn.pool([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS],
                   [conv_size, conv_size, img_CHANNELS, conv_KERNEL_NUM],
                   [[1, length_step, length_step, 1],
                   padding='SAME')
'''

def get_weight(shape, regularizer):
# 去掉偏离太大的正态w
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x, w):
# x = [BATCH_SIZE, 边长， 边长, 通道数] w = [conv_size, conv_size, NUM_CHANNELS, CONV_KERNEL_NUM)]
# 输入图片x 使用卷积核w 步长strides 是否全零填充
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
#    return tf.nn.conv2d(_input, _filter, _strides, _padding)
    
def max_pool_2x2(x):
#    return tf.nn.max_pool(_value, _ksize, _strides, _padding)
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# train 参数是为了判断是否需要 dropout
def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
# 一层一个偏置 故 偏置数 = 卷积核通道数
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)
# 输出shape 在代码中不用人为计算 只用 定义所用卷积核 和 传递上一层输出结果

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)
    
    pool_shape = pool2.get_shape().as_list()
# pool_shape = [BATCH_SIZE, IMG_SiZE, IMG_SIZE, NUM_CHANNELS]
# x 的 BATCH_SIZE 从 conv1 引入一直藏在 shape 里面
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    
# 定义全连接网络 时刻记得矩阵乘法
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1, KEEP_PROB)

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    
    return y
