from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np

class ZicoMNIST(object):
    def __init__(self):
        self.batch_size = 100
        self.learning_rate_base = 0.8
        self.learning_rate_decay = 0.99
        self.ragularatztion_rate = 0.0001
        self.traning_steps = 30000
        self.moving_average_decay = 0.99
        self.model_save_path = "./model"
        self.model_name = "model.ckpt"

    def  train(self, mnist):
      with tf.variable_scope('net'):
        W0 = tf.get_variable('W0', dtype=tf.float32, shape=(4, 4, 1, 16))
        B0 = tf.get_variable('B0', dtype=tf.float32, shape=(16,))
        W2 = tf.get_variable('W2', dtype=tf.float32, shape=(4, 4, 16, 32))
        B2 = tf.get_variable('B2', dtype=tf.float32, shape=(32,))
        W5 = tf.get_variable('W5', dtype=tf.float32, shape=(1568, 100))
        B5 = tf.get_variable('B5', dtype=tf.float32, shape=(100,))
        W7 = tf.get_variable('W7', dtype=tf.float32, shape=(100, 10))
        B7 = tf.get_variable('B7', dtype=tf.float32, shape=(10,))
        x = tf.placeholder(
            tf.float32,[None,28,28,1], name='x-input')
        y_ = tf.placeholder(
            tf.float32,[None,10], name='y-input')
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        y = tf.nn.conv2d(y, W0, strides=[1, 2, 2, 1], padding='VALID')
        y = tf.nn.bias_add(y, B0)
        y = tf.nn.relu(y)
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]])
        y = tf.nn.conv2d(y, W2, strides=[1, 2, 2, 1], padding="VALID")
        y = tf.nn.bias_add(y, B2)
        y = tf.nn.relu(y)
        y = tf.transpose(y, [0, 3, 1, 2])
        y = tf.reshape(y, [tf.shape(y)[0], -1])
        y = np.dot(y, W5 + B5)
        y = tf.nn.relu(y)
        y = np.dot(y, W7 + B7)

        global_step = tf.Variable(0, trainable=False)
        variable_average = tf.train.ExponentialMovingAverage(
            self.moving_average_decay, global_step)
        variable_average_op = variable_average.apply(tf.trainable_variables())
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean 
        learning_rate = tf.train.exponential_decay(
            self.learning_rate_base,
            global_step,
            mnist.train.num_examples/self.batch_size,
            self.learning_rate_decay)
        train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                        .minimize(loss, global_step = global_step)
        with tf.control_dependencies([train_step, variable_average_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            for i in range(self.traning_steps):
                xs, ys = mnist.train.next_batch(self.batch_size)
                _, loss_value, step  = sess.run([train_op, loss, global_step],
                                                feed_dict={x: xs.reshape(100,28,28,1), y_: ys})
                if i % 1000 ==0:
                    print("After %d training steps, loss on training "
                        " batch is %g."%(step, loss_value))
                    saver.save(
                        sess, os.path.join(self.model_save_path,self.model_name),
                        global_step=global_step)
def main(argv=None):
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    zico = ZicoMNIST()
    zico.train(mnist)
if __name__  == '__main__':
    tf.app.run() 