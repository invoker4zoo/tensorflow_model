# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
# import utils
import os

slim = tf.contrib.slim


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


# def get_style_features(endpoints_dict, style_layers):
#     """
#     For the "style_image", the preprocessing step is:
#     1. Resize the shorter side to FLAGS.image_size
#     2. Apply central crop
#     """
#     with tf.Graph().as_default() as g:
#         features = []
#         for layer in style_layers:
#             feature = endpoints_dict[layer]
#             feature = tf.squeeze(gram(feature), [0])  # remove the batch dimension
#             features.append(feature)
#
#         with tf.Session() as sess:
#             return sess.run(features)




def style_loss(endpoints_dict, style_layers, style_path):
    style_loss = 0
    style_loss_summary = {}
    style_gram = tf.squeeze(gram(endpoints_dict[layer]), [0])
    for layer in style_layers:

        generated_images = endpoints_dict[layer]
        size = tf.size(generated_images)
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss
    return style_loss, style_loss_summary
    #
    # for style_gram, layer in zip(style_features_t, style_layers):
    #     generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
    #     size = tf.size(generated_images)
    #     layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
    #     style_loss_summary[layer] = layer_style_loss
    #     style_loss += layer_style_loss
    # return style_loss, style_loss_summary


def content_loss(endpoints_dict, content_layers, content_path):
    content_loss = 0
    for layer in content_layers:
        generated_images = endpoints_dict[layer],
        size = tf.size(generated_images)
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper
    return content_loss, generated_images


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss
