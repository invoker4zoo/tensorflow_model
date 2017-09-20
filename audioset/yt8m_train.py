# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: yt8m_train.py
@ time: $17-9-20 下午3:25
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from yt8m_util import cross_entropy_loss,logistic_regress_model