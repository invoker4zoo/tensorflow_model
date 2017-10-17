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


def get_distort_image(data_dir, batch_size, is_eval=False):
    """
    construct distort image input for graph
    :param data_dir:
    :param batch_size:
    :param is_eval: get training data or evl data
    :return:
    """
    DISTORT_IMAGE_HEIGHT = 24
    DISTORT_IMAGE_WEIGHT = 24
    if not is_eval:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
    for f in filenames:
        if not os.path.exists(f):
            raise ValueError('Failed to find file: ' + f)
    # create a queue for reading file names
    file_name_queue = tf.train.string_input_producer(filenames)
    read_result = read_cifar_file(file_name_queue)
    # get original image
    reading_image = read_result.image
    height = DISTORT_IMAGE_HEIGHT
    weight = DISTORT_IMAGE_WEIGHT

    # distort image
    # get part of the image
    distort_image = tf.random_crop(reading_image, [height, weight, 3])

    # Randomly flip the image horizontally.
    distort_image = tf.image.random_flip_left_right(distort_image)

    # image brightness and contrast
    distort_image = tf.image.random_brightness(distort_image, max_delta=63)
    distort_image = tf.image.random_contrast(distort_image, lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    """
      Linearly scales `image` to have zero mean and unit norm.

      This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
      of all values in image, and
      `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

      `stddev` is the standard deviation of all values in `image`. It is capped
      away from zero to protect against division by 0 when handling uniform images.
    """
    float_image = tf.image.per_image_standardization(distort_image)
    # set label shape
    read_result.label.set_shape([1])
    # set batch queue params
    if not is_eval:
        min_queue_example = int(0.4 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    else:
        min_queue_example = int(0.4 * NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
    return generate_batch_label_batch(float_image, read_result.labl, min_queue_example, batch_size, shuffle=True)