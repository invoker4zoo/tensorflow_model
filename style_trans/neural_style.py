# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechat: whatshowlove
@ software: PyCharm
@ file: neural_style
@ time: 18-4-10
"""

import tensorflow as tf
import reader
from preprocessing import preprocessing_factory
from nets import nets_factory
import losses
import time

tf.app.flags.DEFINE_float("CONTENT_WEIGHT", 5e0, "Weight for content features loss")
tf.app.flags.DEFINE_float("STYLE_WEIGHT", 1e2, "Weight for style features loss")
tf.app.flags.DEFINE_float("TV_WEIGHT", 1e-5, "Weight for total variation loss")
tf.app.flags.DEFINE_string("VGG_MODEL", "pretrained/vgg_16.ckpt", "vgg model params path")
tf.app.flags.DEFINE_list("CONTENT_LAYERS", ["vgg_16/conv3/conv3_3"],
                           "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_list("STYLE_LAYERS", ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2",
                                          "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"],
                           "Which layers to extract style from")
tf.app.flags.DEFINE_string("SUMMARY_PATH", "tensorboard", "Path to store Tensorboard summaries")
tf.app.flags.DEFINE_string("STYLE_IMAGE", "img/picasso.jpg", "Styles to train")
tf.app.flags.DEFINE_float("STYLE_SCALE", 1.0, "Scale styles. Higher extracts smaller features")
tf.app.flags.DEFINE_float("LEARNING_RATE", 10., "Learning rate")
tf.app.flags.DEFINE_string("CONTENT_IMAGE", "img/dancing.jpg", "Content image to use")
tf.app.flags.DEFINE_boolean("RANDOM_INIT", True, "Start from random noise")
tf.app.flags.DEFINE_integer("NUM_ITERATIONS", 1000, "Number of iterations")
# reduce image size because of cpu training
tf.app.flags.DEFINE_integer("IMAGE_SIZE", 256, "Size of output image")


FLAGS = tf.app.flags.FLAGS


def variation_loss(layer):
    """
    图层图像标准差
    :param layer:
    :return:
    """
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.pack([-1,height-1,-1,-1])) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.pack([-1,-1,width-1,-1])) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))


def gram(layer):
    """
    得到输出层的gram矩阵，用于风格误差比较
    :param layer:
    :return:
    """
    shape = tf.shape(layer)
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.pack([-1, num_filters]))
    gram = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(size)

    return gram

def main(argv=None):
    network_fn = nets_factory.get_network_fn(
        'vgg_16',
        num_classes=1,
        is_training=False)
    image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
        'vgg_16',
        is_training=False)

    preprocess_content_image = reader.get_image(FLAGS.CONTENT_IMAGE, FLAGS.IMAGE_SIZE)

    # add bath for vgg net training
    preprocess_content_image = tf.expand_dims(preprocess_content_image, 0)
    _, endpoints_dict = network_fn(preprocess_content_image, spatial_squeeze=False)


    # Log the structure of loss network
    tf.logging.info('Loss network layers(You can define them in "content_layers" and "style_layers"):')
    for key in endpoints_dict:
        tf.logging.info(key)

    """Build Losses"""
    # style_features_t = losses.get_style_features(endpoints_dict, FLAGS.STYLE_LAYERS)
    content_loss, generaged_image = losses.content_loss(endpoints_dict, FLAGS.CONTENT_LAYERS, FLAGS.CONTENT_IMAGE)
    style_loss, style_loss_summary = losses.style_loss(endpoints_dict, FLAGS.style_layers, FLAGS.STYLE_IMAGE)
    tv_loss = losses.total_variation_loss(generaged_image)  # use the unprocessed image

    loss = FLAGS.STYLE_WEIGHT * style_loss + FLAGS.CONTENT_WEIGHT * content_loss + FLAGS.TV_WEIGHT * tv_loss
    train_op = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE).minimize(loss)

    output_image = tf.image.encode_png(tf.saturate_cast(tf.squeeze(generaged_image) + reader.mean_pixel, tf.uint8))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        start_time = time.time()
        for step in range(FLAGS.NUM_ITERATIONS):
            _, loss_t, cl, sl = sess.run([train_op, loss, content_loss, style_loss])
            elapsed = time.time() - start_time
            start_time = time.time()
            print(step, elapsed, loss_t, cl, sl)
        image_t = sess.run(output_image)
        with open('out.png', 'wb') as f:
            f.write(image_t)

if __name__ == '__main__':
    tf.app.run()