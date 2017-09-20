# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: yt8m_util.py
@ time: $17-9-20 下午3:56
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


def find_class_by_name(name, modules):
    """
    get class from provided modules
    :param name: class name
    :param modules: modules list
    :return: class
    """
    class_list = [getattr(module, name, None) for module in modules]
    return next(class_item for class_item in class_list)


##########################
# loss fun
def cross_entropy_loss(predictions, labels):
    """
    build loss tensor
    :param prediction: prediction tensor
    :param labels: labels tensor
    :return: loss tensor
    """
    with tf.name_scope("loss_xent"):
        epsilon = 10e-6
        float_labels = tf.cast(labels, tf.float32)
        cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + \
                             (1 - float_labels) * tf.log(1 - predictions + epsilon)
        cross_entropy_loss = tf.negative(cross_entropy_loss)
        return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


############################
# classify model
def logistic_regress_model(model_input, class_size, l2_params=1e-8):
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

