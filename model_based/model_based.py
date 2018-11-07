import numpy as np
import pickle
import tensorflow as tf
import math

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python import ops

import gym
env = gym.make('CartPole-v0')

learning_rate = 1e-2
gamma = 0.99
decay_rate = 0.99
resume = False


hidden_layer_1 = 8


tf.reset_default_graph()
observations = tf.placeholder(tf.float32, [None, 4], name="input_x")
W1 = tf.Variable(tf.truncated_normal([4, hidden_layer_1],
                                     dtype=tf.float32,
                                     stddev=1e-1))
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, 1],
                                     dtype=tf.float32,
                                     stddev=1e-1))
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)

loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
