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
import time
import os
import sys
import numpy as np


class CharRnn(object):

    def __init__(self, words_size, model='lstm', hidden_size=128, num_layers=2,
                 learning_rate=1.0, keep_prob=1, batch_size=64, seq_size=20, grad_clip=5):
        """

        :param model: model type 组建rnn网络的单元类型
        :param hidden_size: lstm内的节点数
        :param num_layers: lstm节点的堆叠数
        :param learning_rate: 学习速率

        """
        self.words_size = words_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = model
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.grad_clip = grad_clip
        self.sess = tf.Session()
        # initial input tensor
        self.build_inputs()
        # build rnn graph
        # logitis, cell, prediction, initial_state, final_state
        self.build_neural_network()
        # build loss graph
        self.build_loss()
        # build optimizer graph
        # grad using clip_by_global_norm
        self.build_optimizer()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.batch_size, self.seq_size), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.batch_size, self.seq_size), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            with tf.device("/cpu:0"):
                embedding = tf.get_variable('embedding', [self.batch_size, self.hidden_size])
                self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

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
        cell = tf.nn.rnn_cell.MultiRNNCell([cell_item] * self.num_layers, state_is_tuple=True)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        # 通过dynamic_rnn对cell展开时间维度
        self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs,
                                                                initial_state=self.initial_state)
        output = tf.reshape(self.lstm_outputs, [-1, self.hidden_size])
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([self.hidden_size, self.words_size], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.words_size))
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.prediction = tf.nn.softmax(self.logits, name='predictions')


    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.words_size)
            y_reshaped = tf.reshape(y_one_hot, -1)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_step, save_path, log_freq, save_freq):
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            # 初始化state
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_freq == 0:
                    print('step: {}/{}... '.format(step, max_step),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_freq == 0):
                    self.saver.save(sess, os.path.join(save_path, 'rnn.module'), global_step=step)
                if step >= max_step:
                    break
            self.saver.save(sess, os.path.join(save_path, 'rnn.module'), global_step=step)

    def generate(self, gene_size, start_word, word_size, checkpoint=None):
        if checkpoint:
            self.load(checkpoint)
        gene_samples = [c for c in start_word]
        sess = self.sess
        new_state = sess.run(self.initial_state)
        prediction = np.ones((word_size,))
        for c in start_word:
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {
                self.inputs: x,
                self.keep_prob: 1,
                self.initial_state: new_state
            }
            prediction, new_state = sess.run([self.prediction, self.final_state],
                                       feed_dict=feed)
        c = random_pick_top_n(prediction, word_size)
        gene_samples.append(c)
        for i in range(gene_size):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {
                self.inputs: x,
                self.keep_prob: 1,
                self.initial_state: new_state
                }
            prediction, new_state = sess.run([self.prediction, self.final_state],
                                             feed_dict=feed)
            c = random_pick_top_n(prediction, word_size)
            gene_samples.append(c)
        return np.array(gene_samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

def random_pick_top_n(preds, word_size, top_n=5):
    """
    从概率预测preds中取出top_n的字典标号，随机取一个编号返回字符编号
    :param preds:
    :param vocab_size:
    :param top_n:
    :return:
    """
    # np.squeeze 删除维度为1的维度
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(word_size, 1, p=p)[0]
    return c