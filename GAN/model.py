# coding=utf-8

from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
from skimage.io import imsave
from tensorflow.examples.tutorials.mnist import input_data


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
        self.batch_size = 256
        self.FILE_PATH = '/home/showlove/code/tensorflow_model/tmp/MNIST_data'

        # class init
        self.__load_data()

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
        w2 = tf.Variable(tf.truncated_normal([self.hidden_layer_2, self.hidden_layer_1],stddev=0.1), name="d_w2", dtype=tf.float32)
        b2 = tf.Variable(tf.zeros([self.hidden_layer_1], stddev=0.1), name="d_b2", dtype=tf.float32)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    def build_graph(self):
        pass

    def load_data(self):
        pass
        # b2 = tf.Variable(tf.zeros([self.hidden_layer_1]), name="d_b2", dtype=tf.float32)
        # h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2),keep_prob)
        # w3 = tf.Variable(tf.truncated_normal([self.hidden_layer_1, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
        # b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
        # h3 = tf.matmul(h2, w3) + b3
        # y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [self.batch_size, -1], name=None))
        # y_generated = tf.nn.sigmoid(tf.slice(h3, [self.batch_size, 0], [-1, -1], name=None))
        # d_params = [w1, b1, w2, b2, w3, b3]
        # return y_data, y_generated, d_params

    def build_graph(self):
        # init graph
        x_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size], name='x_input')
        z_prior = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_size], name='z_prior')
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # 创建生成模型
        x_generated, g_params = self.generator(z_prior)
        # 创建判别模型
        y_data, y_generated, d_params = self.discriminator(x_input, x_generated, keep_prob)

        # 损失函数的设置
        # y_generated=D(G(z)) y_data=D(x)
        #
        d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
        g_loss = - tf.log(y_generated)

        optimizer = tf.train.AdamOptimizer(0.0001)

        # 两个模型的优化函数
        d_trainer = optimizer.minimize(d_loss, var_list=d_params)
        g_trainer = optimizer.minimize(g_loss, var_list=g_params)

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()

        # 启动会话
        with tf.Session() as sess:
            # 初始化变量
            sess.run(init)


    def __load_data(self):
        self.mnist = input_data.read_data_sets(self.FILE_PATH, one_hot=True)

    def _model_saving(self):
        pass

    def _data_saving(self):
        pass

    def saving(self):
        pass

    def show_result(self, batch_res, fname, grid_size=(8, 8), grid_pad=5):
        batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], self.img_height, self.img_width)) + 0.5
        img_h, img_w = batch_res.shape[1], batch_res.shape[2]
        grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
        grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
        img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        for i, res in enumerate(batch_res):
            if i >= grid_size[0] * grid_size[1]:
                break
            img = (res) * 255
            img = img.astype(np.uint8)
            row = (i // grid_size[0]) * (img_h + grid_pad)
            col = (i % grid_size[1]) * (img_w + grid_pad)
            img_grid[row:row + img_h, col:col + img_w] = img
        imsave(fname, img_grid)
