# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: yt8m_model.py
@ time: $17-9-20 上午11:33
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


class LogisticRegressModel(object):


    def build_model(self, model_input, class_size, l2_params=1e-8):
        """

        :param model_input: shape is batch * feature_size
        :param class_size:  total class size, as output size
        :param l2_params: l2_regularization params
        :return: output graph
        """
        output = slim.fully_connected(
            model_input, class_size, activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(l2_params)
        )
        return output