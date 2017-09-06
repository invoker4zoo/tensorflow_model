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
import numpy as tf
import time

# gobal params
max_step = 3000
batch_size = 128
data_dir = '../tmp/cifar_data/cifar-10-batches-bin'

image_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
