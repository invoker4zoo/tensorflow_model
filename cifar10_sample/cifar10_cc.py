# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: cifar10_cc.py
@ time: $17-9-6 上午11:55
"""

import tensorflow as tf
import cifar10,cifar10_input
import numpy as np
import time


# function
def variable_with_weight_loss(shape, stddev, wd):
    """
    产生计算L2正则的神经网络权重参数,L2 损失记录在collection中
    :param shape:
    :param stddev:
    :param wd:
    :return: tf.Variable
    """
    var = tf.Variable(tf.truncatedi_normal(shape, stddev))
    if wd is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_L2_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def loss(logits, labels):
    """
    计算损失函数，需要将logits 构建softmax计算后计算交叉熵
    损失函数需要加上全连接层权重系数的L2正则化
    :param logits:
    :param labels:
    :return:
    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example'
                                    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy_mean')


# gobal params
max_step = 3000
batch_size = 128
data_dir = '../tmp/cifar_data/cifar-10-batches-bin'

# get image data from cifar10_input
image_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
image_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# define network graph
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# ####################### CNN PART
# conv layer 1
weight_1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=0.02, wd=0.0)
kernel_1 = tf.nn.conv2d(image_holder, weight_1, strides=[1, 1, 1, 1], padding='SAME')
bias_1 = tf.Variable(tf.constant(0.0, shape=[64]))
# tf.nn.bias_add tf.add
# tf.nn.bias_add是tf.add的一种特殊情况，可以允许bias 和 value 的type不一致，会自动转换
conv_1 = tf.nn.relu(tf.nn.bias_add(kernel_1, bias_1))
pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# LRN local response normalization
norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

# conv layer 2
weight_2 = variable_with_weight_loss(shape=[5,5,64,64], stddev=0.02, wd=0.0)
kernel_2 = tf.nn.conv2d(norm_1, weight_2, strides=[1, 1, 1, 1], padding='SAME')
bias_2 = tf.Variablte(tf.constant(0.1, shape=[64]))
conv_2 = tf.nn.relu(tf.nn.bias_add(kernel_2, bias_2))
norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')

# fully connection layer 3
# 全连接层为了防止过拟合需要对weight加上L2正则
reshape = tf.reshape(pool_2,[batch_size, -1])
dim = reshape.get_shape()[1].value
weight_3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wd=0.004)
bias_3 = tf.Variable(tf.constant(0.1,shape=[384]))
local_3 = tf.nn.relu(tf.matmul(reshape, weight_3) + bias_3)

# fully connection layer 4
weight_4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wd=0.004)
bias_4 = tf.Variable(tf.constant(0.1, shape=[192]))
local_4 = tf.nn.relu(tf.matmul(local_3, weight_4) + bias_4)

# logits layer 5
weight_5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wd=0.0)
bias_5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.nn.bias_add(tf.matmul(local_4, weight_5), bias_5)

# ###########LOSSING FUNCTION
