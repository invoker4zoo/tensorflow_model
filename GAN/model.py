# coding=utf-8

from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np


class DCGan(object):
    def __init__(self):
        pass

    def train(self):
        pass

    def generator(self):
        pass

    def discriminator(self):
        pass

    def build_graph(self):
        pass

    def load_data(self):
        pass

    def _model_saving(self):
        pass

    def _data_saving(self):
        pass

    def saving(self):
        pass


class SampleNNGan(object):
    def __init__(self, config):
        self.config = config

        self.hidden_layer_1 = 150
        self.hidden_layer_2 = 300
        self.z_size = 100

        # using config setting later
        self.img_height = 28
        self.img_width = 28
        self.img_size = self.img_height * self.img_width

    def train(self):
        pass

    def generator(self, z_input):

        w1 = tf.Variable(tf.truncated_normal([self.z_size, self.hidden_layer_1], stddev=0.1), name="g_w1", dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([self.hidden_layer_1]), name="g_b1", dtype=tf.float32)
        h1 = tf.nn.relu(tf.matmul(z_input, w1) + b1)
        w2 = tf.Variable(tf.truncated_normal([self.hidden_layer_1, self.hidden_layer_2], stddev=0.1), name="g_w2", dtype=tf.float32)
        b2 = tf.Variable(tf.zeros([self.hidden_layer_2]), name="g_b2", dtype=tf.float32)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        w3 = tf.Variable(tf.truncated_normal([self.hidden_layer_2, self.img_size], stddev=0.1), name="g_w3", dtype=tf.float32)
        b3 = tf.Variable(tf.zeros([self.img_size]), name="g_b3", dtype=tf.float32)
        h3 = tf.matmul(h2, w3) + b3
        x_generate = tf.nn.tanh(h3)
        g_params = [w1, b1, w2, b2, w3, b3]
        return x_generate, g_params

    def discriminator(self,x_data, x_generated, keep_prob):

        x_input = tf.concat([x_data,x_generated],axis=0)
        w1 = tf.Variable(tf.truncated_normal([self.img_size,self.hidden_layer_2],stddev=0.1), name="d_w1", dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([self.hidden_layer_2]), name="d_b1", dtype=tf.float32)
        h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_input, w1)+b1), keep_prob)
        w2 = tf.Variable(tf.truncated_normal([self.hidden_layer_2,self.hidden_layer_1],stddev=0.1), name="d_w2", dtype=tf.float32)

    def build_graph(self):
        pass

    def load_data(self):
        pass

    def _model_saving(self):
        pass

    def _data_saving(self):
        pass

    def saving(self):
        pass