#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:58:39 2019

@author: yaoxinzhi
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io

'''
复现已有网络 实现特定应用

x = tf.placeholeder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGGE_SIZE, NUM_CHANNELS])
仅占位

np.load() / np.save() 将数组以二进制格式 读出/写入 磁盘 拓展名为 .np
.item() 遍历数组内键值对

tf.shape(a)  返回a的维度 

tf.nn.bias(乘加和, bias)
tf.reshape(tensor, [n行, m列]) -1 表示跟随m列自动计算

np.argsort(列表) 对列表从小到大排序 返回索引值
os.getwd() 返回当前工作目录
os.path.join() 拼出整个路径 可引导到特定文件

tf.split(who, [切分方式], 维度)
tf.concat(value, 维度)

http://tensorflow.google.cn/

fig = plt.figure('pic_name') 实例化图片对象
img = io.imread('pic_path') 读入图片
ax = fig.add_subplot(num, num, num) 包含几行 几列子图 当前是第几个
ax.bar(num, value, name, len, col) 柱状图 （个数 名字 宽度 颜色）
ax.set_ylabel('') y轴名字 u"中文"
ax.set_title('')    子图添加名字
ax.text(x坐标, y坐标， 文字内容, ha='venter',va='bottom',fontsize=7)
ax = imshow('pic')
'''




