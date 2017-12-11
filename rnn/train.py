# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: train.py
@ time: $17-11-3 下午12:15
"""

import tensorflow as tf
from model import CharRnn
from rnn_reader import batch_generator, TextConverter
import os
# codecs为文件读写库，可以指定一种编码方式打开文件，返回unicode，写入时也为指定编码
import codecs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_file', 'data/poetry.txt', 'input txt file')
tf.flags.DEFINE_string('model_path', 'model/poetry/', 'saving path of model and words dict')
tf.flags.DEFINE_string('word_dict_name', 'converter.pkl', 'words dictionary file name')
tf.flags.DEFINE_string('model_type', 'lstm', 'rnn model cell type')
tf.flags.DEFINE_integer('batch_size', 32, 'training batch size')
tf.flags.DEFINE_integer('seq_size', 26, 'string seq length in one input example')
tf.flags.DEFINE_integer('hidden_size', 128, 'rnn hidden cell unit size')
tf.flags.DEFINE_integer('num_layers', 2, 'num of rnn layers')
tf.flags.DEFINE_integer('max_step', 100000, 'the max training steps')
tf.flags.DEFINE_integer('save_freq', 1000, 'save model every steps during training')
tf.flags.DEFINE_integer('log_freq', 100, 'write log every steps during training')
tf.flags.DEFINE_integer('max_dict_size', 5000, 'the max size of the words dict')
tf.flags.DEFINE_float('learning_rate', 0.005, 'optimizer learning rate')
tf.flags.DEFINE_float('keep_prob', 0.5, 'drop out prob during training')


def main(_):
    if not os.path.exists(FLAGS.model_path):
        os.mkdir(FLAGS.model_path)
    with codecs.open(FLAGS.input_file, 'r', encoding='utf8') as f:
        text = f.read()
    converter = TextConverter(text, max_vocab=FLAGS.max_dict_size)
    converter.save_to_file(os.path.join(FLAGS.model_path, FLAGS.word_dict_name))
    arr = converter.text_to_arr(text)
    g = batch_generator(arr, batch_size=FLAGS.batch_size, seq_size=FLAGS.seq_size)
    model = CharRnn(converter.vocab_size,
                    model=FLAGS.model_type,
                    hidden_size=FLAGS.hidden_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    keep_prob=FLAGS.keep_prob,
                    batch_size=FLAGS.batch_size,
                    seq_size=FLAGS.seq_size,
                    # 是否为训练模型,batch_size 和seq_size 会不一样
                    is_train=True,
                    using_embedding=False
                    )
    model.train(g,
                max_step=FLAGS.max_step,
                save_path=FLAGS.model_path,
                log_freq=FLAGS.log_freq,
                save_freq=FLAGS.save_freq
                )


if __name__ == '__main__':
    tf.app.run()