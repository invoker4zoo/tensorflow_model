# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: cifar10_cc_input.py
@ time: $17-10-11 上午9:55
"""
import tensorflow as tf
import os

# set image size ## 32/24
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar_file(file_name_queue):
    """
    read data using tf reader, trans data type ,structure image and label data
    :param file_name_queue: tf filename queue
    :return: result class
    contain image data and label data
    """
    class resultStructure(object):
        pass
    result = resultStructure()
    label_bytes = 1
    result.height = 32
    result.weight = 32
    result.channel = 3
    image_bytes = result.height * result.weight * result.channel
    result_bytes = label_bytes + image_bytes
    # initial reader
    # read bytes from file
    reader = tf.FixedLengthRecordReader(record_bytes=result_bytes)
    result.key, value = reader.read(file_name_queue)
    # trans value type / slice image data and label data
    # convert string to a tensor
    result_bytes = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(tf.strided_slice(result_bytes, 0, [label_bytes]), tf.int32)
    # the origin image data is 3072 bytes, trans to 3 * 32 * 32 tensor
    image_origin = tf.reshape(
                        tf.strided_slice(result_bytes, [label_bytes], [label_bytes + image_bytes]),
                        [result.channel, result.height, result.weight]
                    )
    # convert [channel * weight * height] to [weight * height * channel]
    result.image = tf.cast(tf.transpose(image_origin, [1, 2, 0]), tf.float32)
    return result


def generate_batch_label_batch(image, label, min_queue_examples, batch_size, shuffle=True):
    """
    build data batch for training or evl
    :param image:
    :param label:
    :param min_queue_examples:
    :param batch_size:
    :param shuffle:
    :return:
    """
    num_multi_threadings = 16
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            num_threads=num_multi_threadings,
            min_after_dequeue=min_queue_examples
        )
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            num_threads=num_multi_threadings
        )
    tf.summary.image('images', image_batch)

    return image_batch, tf.reshape(label_batch, [batch_size])


def distort_iimage(data_dir, batch_size):
    """
    construct distort image input
    :param data_dir:
    :param batch_size:
    :return:
    """
    pass