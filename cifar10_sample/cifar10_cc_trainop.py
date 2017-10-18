# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: cifar10_cc_trainop.py
@ time: $17-10-13 下午4:25
"""

import tensorflow as tf
import cifar10_cc_input
import re
import os


# golbal params
TOWER_NAME = 'tower'
USE_FP16 = True
BATCH_SIZE = 128
DATA_DIR = 'home/showlove/cc/code/tensorflow_model/tmp/cifar10_data'
TRAIN_DIR = ''
NUM_CLASS = cifar10_cc_input.NUM_CLASSES
IMAGE_SIZE = cifar10_cc_input.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_cc_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_cc_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def tensor_summary(x):
    """
    create tensor histogram and scalar
    remove tower_[0-9] for using multi gpu
    :param x:
    :return:
    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    # tf.nn.zero_traction(tensor) return the zero fraction of tensor 0的占比
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def variable_on_cpu(name, shape, initializer):
    """
    create variable on cpu
    :param name: variable name
    :param shape:
    :param initializer:
    :return:
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if USE_FP16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def get_variable_with_l2(name, shape, stddev, wd):
    """
    add l2 loss in connection ,cal total loss
    :param name:
    :param shape:
    :param stddev: trunc_normal params stddev
    :param wd: l2 loss wright
    :return: var and add l2 loss in connection
    """
    dtype = tf.float16 if USE_FP16 else tf.float32
    var = variable_on_cpu(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        loss = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', loss)
    return var


def distort_input():
    """
    construct distort input from data_dir
    using cifar_input_cc get_distort_image fun
    :return: batch of images, labels
    """
    if not os.path.exists(DATA_DIR):
        raise ValueError('data path is not existing')
    data_dir = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
    images, labels = cifar10_cc_input.get_distort_image(data_dir=data_dir, batch_size=BATCH_SIZE)
    if USE_FP16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def model(images):
    """
    :param images:
    :return: predict result
    the net uses 2 convolution layer + 2 fully connect layer + logistical layer
    the conv layer uses conv + max_pooling + lrn
    """
    # conv1
    with tf.variable_scope('conv1') as scope:
        weight = get_variable_with_l2('weight', [5, 5, 3, 64], stddev=5e-2, wd=0)
        kenerl = tf.nn.conv2d(images, weight, strides=[1, 1, 1, 1], padding='SAME')
        bias = variable_on_cpu('bias', shape=[64], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.relu(tf.nn.bias_add(kenerl, bias), scope.name)
        tensor_summary(conv1)
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    # conv2
    with tf.variable_scope('covn2') as scope:
        weight = get_variable_with_l2('weight',[5, 5, 64, 64], stddev=5e-2, wd=0)
        kenerl = tf.nn.conv2d(norm1, weight, strides=[1, 1, 1, 1], padding='SAME')
        bias = variable_on_cpu('bias', shape=[64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.relu(tf.nn.add_bias(kenerl, bias), scope.name)
        tensor_summary(conv2)
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    # local 3
    with tf.variable_scope('local3') as scope:
        reshape_layer = tf.reshape(norm2, [BATCH_SIZE, -1])
        dim = reshape_layer.get_shape()[1]
        weight = get_variable_with_l2('weight', [dim, 384], stddev=0.04, wd=0.004)
        bias = variable_on_cpu('bias', shape=[384], initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool2, weight), bias), name=scope.name)
        tensor_summary(local3)

    # local 4
    with tf.variable_scope('local4') as scope:
        weight = get_variable_with_l2('weight', [384, 192], stddev=0.04, wd=0.4)
        bias = variable_on_cpu('bias', shape=[192], initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(local3, weight), bias),name=scope.name)
        tensor_summary(local4)

    # logits
    with tf.variable_scope('logits') as scope:
        weight = get_variable_with_l2('weight', [192, NUM_CLASS], stddev=1/192.0, wd=0.0)
        bias = variable_on_cpu('bias', [NUM_CLASS], initializer=tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(local4, weight), bias, name=scope.name)
        tensor_summary(logits)
    return logits


def loss(logits, labels):
    """
    calculate total loss
    :param logits:
    :param labels:
    :return:
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # get total losses from collection
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def loss_moving_avg_summary(total_loss):
    """
    calculate moving avg of all kinds og losses, add to summary
    :param total_loss:
    :return:
    """
    loss_avg_cal = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_avg_cal.average()
    losses_list = tf.get_collection('losses')
    losses_op = loss_avg_cal.apply(losses_list + [total_loss])
    # attach summary to all kinds of losses
    for loss_item in losses_list + [total_loss]:
        tf.summary.scalar(loss_item.name + '_origin', loss_item)
        tf.summary.scalar(loss_item.name, loss_avg_cal.average(loss_item))
    return losses_op


def train(total_loss, global_step):
    """
    construct train op
    :param total_loss:
    :param golbal_step:
    :return:
    """
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = loss_moving_avg_summary(total_loss)

    # Compute gradients.
    # with tf.control_dependencies
    # in session calculate loss_averages_op first
    # after then cal grads
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        # opt.compute_gradients return grads_and_vars
        grads = opt.compute_gradients(total_loss)

    # Apply gradients. params grads_and_vars /get from opt.compute_gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients. only calculate total_loss
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op