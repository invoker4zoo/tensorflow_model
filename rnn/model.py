# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: model.py
@ time: $17-10-31 下午3:55
"""

import tensorflow as tf


class CharRnn(object):

    def __init__(self, model='lstm', hidden_size=128, num_layers=2, learning_rate=1.0, keep_prob=1):
        """

        :param model: model type 组建rnn网络的单元类型
        :param hidden_size: lstm内的节点数
        :param num_layers: lstm节点的堆叠数
        :param learning_rate: 学习速率

        """
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = model
        self.keep_prob = keep_prob

    def build_neural_network(self, is_train=True):
        if self.cell_type == 'rnn':
            cell_fun = tf.nn.rnn_cell.BasicRNNCell
        elif self.cell_type == 'gru':
            cell_fun = tf.nn.rnn_cell.GRUCell
        elif self.cell_type == 'lstm':
            cell_fun = tf.nn.rnn_cell.BasicLSTMCell

        if is_train and self.keep_prob < 1:
            cell_item = tf.nn.rnn_cell.DropoutWrapper(cell_fun(self.hidden_size, forget_bias=0.0, state_is_tuple=True),
                                                      output_keep_prob=self.keep_prob)
        else:
            cell_item = cell_fun(self.hidden_size, forget_bias=0.0, state_is_tuple=True)

