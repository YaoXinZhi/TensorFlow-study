#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:35:40 2019

@author: yaoxinzhi
"""

'''
MNIST 数据集
6+1 w 张 手写数字
每张图784个像素点 6w张给出 数值出现概率

from tensorflow.exaamples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/', one_hot=True)
Train / validation / Test 三个子集

返回各子集数目
mnist.train.num_examples
mnist.validation.num_examples
mist.test.num_examples

mnist.train.labels[0] 查看指定图标签
mnist.train.images[0] 查看制定图像素点

取出指定数据集
BATCH_SIZE = 200
xs, ys = mnist.train.next_batch(BATCH_SIZE)
xs.shape 200*784
ys.shape 200*10

tf.get_collection("") 从集合中取出全部变量 生成一个列表
tf.add_n([]) 列表内对应元素相加
tf.cast(x, dtype) 把x转为dtype 类型
tf.argmax(x, axis) 返回axis指定维度中最大值所在索引号 例如 tf.argmax([1, 0, 0], 1) 返回 0
os.path.join("home", "name") 返回home/name
with tf.Graph().as_default() as g: 其内定义的节点在计算图g中

在反向传播中 保存模型
saver = tf.train.Saver() 实例化saver对象
with tf.Session() as sess:
    for i in range(STEPS):
        if i%轮数 == 0：
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
    
加载模型
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(储存路径)
    if ckpt and ckpt_model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        
实例化可还原滑动平均值的saver
ema = tf.train.ExponentialMovingAverage(滑动平均值)
ema_restore = ema.variables_to_restore()
saver = tf.train.Saver(ema_restore)

准确率计算方法
y BATCH_SIZE*10
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuray = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''