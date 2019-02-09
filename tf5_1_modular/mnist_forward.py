#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 21:44:41 2019

@author: yaoxinzhi
"""
'''
MNIST 数据集 
http://yann.lecun.com/exdb/mnist/ 手动下载数据集到 './MNIST_data/'

from tensorflow.examples.tutorials.mnist import input_data
mnist = imput_data.read_data_sets('./MNIST_data/', one_hot=True)
'''
import tensorflow as tf

# 每张图 784 个像素点 一维数组
INPUT_NODE = 784
# 输出 10 分类
OUTPUT_NODE = 10
# 第一个隐藏层节点数
LAYER1_NODE = 500

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    
    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias(OUTPUT_NODE)
    y = tf.matmul(y1, w2) + b2
    
    return y