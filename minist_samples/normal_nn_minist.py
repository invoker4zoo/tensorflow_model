#coding=utf-8
import input_data
import tensorflow as tf
import os


'''
  mnist 数据集,one_hot = True 标签做独热编码
  包括三个部分
  data_sets.train
  data_sets.validation
  data_sets.test
'''
mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=True)

"""
构建有一层隐含层，10个节点的简单神经网络
"""
x = tf.placeholder('float', [None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y_hat = tf.nn.softmax(tf.matmul(x,W)+b)

"""
交叉熵计算
描述预测值的低效性
损失函数
y为实际的概率分布，y_hat为预测的概率分布
"""

y = tf.placeholder('float',[None,10])
cross_entropy = -tf.reduce_sum(y*tf.log(y_hat))

"""
损失函数训练
"""
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

"""
评价函数
"""
correct_prediction = tf.equal(tf.argmax(y,1),tf.arg_max(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))


"""
模型存储测试
"""
saver = tf.train.Saver()

"""
使用会话进行训练
"""
# 初始化所有的变量
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
    # saver
    save_path = saver.save(sess,save_path='save/model.ckpt')
    print 'save model to %s'%save_path
    print sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})


