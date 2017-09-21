# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: yt8m_reader.py
@ time: $17-9-15 下午3:21
"""

import tensorflow as tf
import numpy as np



#  file reader
class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class YT8MVideoLevelFeatureReader(BaseReader):
    """
    read video-level youtube-8m data set
    in this level set video and audio have embedding feature with 1024D and 128D
    """

    def __init__(self,
                 num_classes=4716,
                 feature_sizes=[128],
                 feature_names=["audio"]):
        """
        build video-level data set reader class
        :param num_classes:
        :param feature_sizes:
        :param feature_names:
        """
        assert len(feature_names) == len(feature_sizes), \
            "length of feature_names (={}) != length of feature_sizes (={})".format( \
                len(feature_names), len(feature_sizes))
        self.num_class = 4716
        self.feature_sizes = feature_sizes
        self.feature_names = feature_names

    def get_batch_data(self, filename_queue, batch_size=1024):

        return self._get_batch_data(filename_queue, batch_size)

    def _get_batch_data(self, filename_queue, batch_size=1024):
        """
        initial reader
        :param filename_queue:
        :param batch_size:
        :return:
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read_up_to(filename_queue, batch_size)
        tf.add_to_collection('serialized_example', serialized_example)
        return self.prepare_serialized_examples(serialized_example)

    def prepare_serialized_examples(self, serialized_example):
        """
        build the input tensor from serialized_example
        :param serialized_example:
        :return:
        """
        num_features = len(self.feature_names)
        assert num_features > 0, "self.feature_names is empty!"
        assert len(self.feature_names) == len(self.feature_sizes), \
            "length of feature_names (={}) != length of feature_sizes (={})".format( \
                len(self.feature_names), len(self.feature_sizes))
        feature_map = {
            'video_id': tf.FixedLenFeature([], tf.string),
            'labels': tf.VarLenFeature(tf.int64)
        }
        for feature_index in range(num_features):
            feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
                self.feature_sizes[feature_index], tf.float32
            )
        features = tf.parse_example(serialized_example, features=feature_map)
        labels = tf.sparse_to_indicator(features["labels"], self.num_class)
        labels.set_shape([None, self.num_class])
        concatenated_features = tf.concat([
                                              features[feature_name] for feature_name in self.feature_names], 1)

        return features["video_id"], concatenated_features, labels, tf.ones([tf.shape(serialized_example)[0]])