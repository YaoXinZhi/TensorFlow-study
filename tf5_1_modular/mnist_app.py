#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:13:24 2019

@author: yaoxinzhi
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward


'''
如何对输入的真实图片 输出预测结果
如何制作数据集 实现特定应用
'''
'''
图片预处理 符合训练好的网络的接口
预测
def applicatoin():
    testNum = input('imput the number of test pictures')
    for i in range(testNum):        
        testPic = raw_input('the path of test picture:')
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print ('The prediction number is:', preValue)
'''

def restore_model(testPicArr):
    with tf.Graph() as g:
# 重现计算图结构
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        # 定义预测结果
        preValue = tf.argmax(y, 1)
        
# 实例化可还原滑动平均值的saver 这样所有参数在被加载时会被赋值为各自的滑动平均值
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        
        with tf.Session() as sess:
# 加载 ckpt
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print ('No checkpoint file found')
                return -1
def pre_pic(picName):
    img = Image.open(picName)
# resize 像素尺寸 消除锯齿
    reIm = img.resize((28, 28), Image.ANTALAS)
# 变为灰度图 并转换为矩阵
    im_arr = np.array(reIm.convert('L'))
    
# 模型要求黑底白字 图片为白底黑字 反色 让图片只有纯黑色 纯白色点 过滤噪声
# threshold 为 黑白色阈值 可更改
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
# 更改形状
    nm_arr = im_arr.reshape([1, 784])
# 模型要求像素点为 0 到 1 之间浮点数
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)
    
    return img_ready

def application():
# 从控制台读入数字
    testNum = int(input('input the number of test pictures:'))
    for i in range(testNum):
# 从控制台读入字符串
        testPic = input('the path of test picture:')
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print ('The prediction number is: {0}'.format(preValue))

def main():
    application()
    
if __name__ == '__main__':
    main()