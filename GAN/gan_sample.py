# coding = utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os

flags = tf.app.flags
#
flags.DEFINE_integer('img height', 28, "input image height")
flags.DEFINE_integer('img weight', 28, "input image weight")
flags.DEFINE_integer('batch size', 256, "num of img size for each batch")
flags.DEFINE_integer('epoch', 100, "epoch for training")

flags.DEFINE_float('learning_rate', 0.0002, "Learning rate for adam")

flags.DEFINE_string('checkpoint_dir', "checkpoint", "Directory name for saving the checkpoints")
flags.DEFINE_string('sample_dir', "samples", "Directory name for saving samples")