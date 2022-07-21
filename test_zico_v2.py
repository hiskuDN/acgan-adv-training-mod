"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


class ZicoMNIST(object):
    def __init__(self, images):
        self.x_image = images
        W0 = tf.compat.v1.Variable('W0', dtype=tf.float32, shape=(4, 4, 1, 16))
        B0 = tf.compat.v1.Variable(dtype=tf.float32, shape=(16,))
        W2 = tf.compat.v1.Variable(dtype=tf.float32, shape=(4, 4, 16, 32))
        B2 = tf.compat.v1.Variable(dtype=tf.float32, shape=(32,))
        W5 = tf.compat.v1.Variable(dtype=tf.float32, shape=(1568, 100))
        B5 = tf.compat.v1.Variable(dtype=tf.float32, shape=(100,))
        W7 = tf.compat.v1.Variable(dtype=tf.float32, shape=(100, 10))
        B7 = tf.compat.v1.Variable(dtype=tf.float32, shape=(10,))

        y = tf.pad(tensor=self.x_image, paddings=[
                   [0, 0], [1, 1], [1, 1], [0, 0]])
        y = tf.nn.conv2d(input=y, filters=W0, strides=[
                         1, 2, 2, 1], padding='VALID')
        y = tf.nn.bias_add(y, B0)
        y = tf.nn.relu(y)
        y = tf.pad(tensor=y, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        y = tf.nn.conv2d(input=y, filters=W2, strides=[
                         1, 2, 2, 1], padding="VALID")
        y = tf.nn.bias_add(y, B2)
        y = tf.nn.relu(y)
        y = tf.transpose(a=y, perm=[0, 3, 1, 2])
        y = tf.reshape(y, [tf.shape(input=y)[0], -1])
        y = y @ W5 + B5
        y = tf.nn.relu(y)
        y = y @ W7 + B7

        self.logits = y
