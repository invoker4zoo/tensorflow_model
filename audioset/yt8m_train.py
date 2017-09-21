# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: yt8m_train.py
@ time: $17-9-20 下午3:25
"""
import tensorflow as tf
# import tensorflow.contrib.slim as slim
from tensorflow import gfile
from tensorflow import logging
from yt8m_util import cross_entropy_loss,logistic_regress_model
from yt8m_reader import YT8MVideoLevelFeatureReader
import time

batch_size = 1024
# data_dir = '/tmp/audioset_v1_embedings/unbal_train'
data_dir = '/home/showlove/cc/youtube-8m/tmp/audioset_v1_embeddings/unbal_train'
train_dir = '/tmp/yt8m_model'
class_size = 4716
max_epoch = 100
max_step = 10000

files = gfile.Glob(data_dir)
filename_queue = tf.train.string_input_producer(
    files, num_epochs=None, shuffle=True)
reader = YT8MVideoLevelFeatureReader(num_classes=class_size)
unused_id, input_feature, lables, num_frames = reader.get_batch_data(filename_queue)
unused_id, input_feature, lables, num_frames = tf.train.shuffle_batch([unused_id, input_feature, lables, num_frames],
                                                                    batch_size=batch_size,
                                                                    capacity=5*batch_size,
                                                                    min_after_dequeue=batch_size,
                                                                    allow_smaller_final_batch=True,
                                                                    enqueue_many=True
                                                                    )
y_out = logistic_regress_model(input_feature, class_size)
loss = cross_entropy_loss(y_out, lables)
train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # tf.train.start_queue_runners()
    for epoch in range(1, max_epoch):
        epoch_begin_time = time.time()
        feature_bath, labels_batch = sess.run([input_feature, lables])
        for step in range(1, max_step):
            _, training_loss = sess.run([train_op, loss], feed_dict={input_feature:feature_bath, lables:labels_batch})
        epoch_loss = training_loss
        epoch_time = time.time() - epoch_begin_time
        print 'epoch %d - training loss %f'%(epoch, epoch_loss)