# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: generate
@ time: 17-11-21
"""

import tensorflow as tf
from model import CharRnn
from rnn_reader import TextConverter
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', 'model/poetry/converter.pkl', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'model/poetry/', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '林玲', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 72, 'max length to generate')


def main(_):
    FLAGS.start_string = FLAGS.start_string.decode('utf-8')
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRnn(words_size=converter.vocab_size,
                    hidden_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    is_train=False
                    )

    model.load(FLAGS.checkpoint_path)

    start = converter.text_to_arr(FLAGS.start_string)
    arr = model.generate(FLAGS.max_length, start, converter.vocab_size)
    print(converter.arr_to_text(arr))


if __name__ == '__main__':
    tf.app.run()