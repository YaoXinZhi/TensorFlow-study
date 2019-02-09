#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 22:27:47 2019

@author: yaoxinzhi
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward

# 程序循环的间隔时间
TEST_INTERVAL_SECS = 5

def test(mnist):

# 其内定义的节点在计算图g中
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
# 测试不用正则化
        y = mnist_forward.forward(x, None)

# 实例化可还原滑动平均值的saver 这样所有参数在被加载时会被赋值为各自的滑动平均值
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
# 准确率计算
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        while True:
            with tf.Session() as sess:
# 加载 ckpt 就是把滑动平均值赋给各个参数
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print ('After {0} training step(s), test accuracy is {1}'.format(global_step, accuracy_score))
                else:
                    print ('No checkpoint file found')
                    return 
                time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
    test(mnist)
    
if __name__ == '__main__':
    main()
                    