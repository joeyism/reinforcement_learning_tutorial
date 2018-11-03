import gym
import numpy as np
import tensorflow as tf
import random

env = gym.make('FrozenLake-v0')

def gen_one_hot(index):
    one_hot_observation = np.zeros(16)
    one_hot_observation[index] = 1
    return one_hot_observation

tf.reset_default_graph()

##### MODEL #####
learning_rate = 0.001
input_placeholder = tf.placeholder(tf.float32, shape=[1, 16])
W = tf.Variable(tf.random_uniform([16,4]))
Qout = tf.matmul(input_placeholder, W)
predict = tf.argmax(Qout,1)

target_Q = tf.placeholder(tf.float32, shape=[1, 4])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=Qout,
        labels=target_Q))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#################

y = 0.99
threshold = 0.1
num_episodes = 2000
val = None

stepList = []
rewardList = []

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(num_episodes):
        observation = env.reset()
        total_reward = 0
        done = False
        step = 0
        while step < 99:
            step += 1

            action, Qout_value = sess.run([predict, Qout],
                    feed_dict={
                        input_placeholder: [gen_one_hot(observation)]
                        })
            action = action[0]

            if np.random.randn(1) < threshold:
                action = env.action_space.sample()

            new_observation, reward, done, _ = env.step(action)
            new_Q = sess.run(Qout, feed_dict={input_placeholder: [gen_one_hot(new_observation)]})

            max_new_Q = np.max(new_Q)
            targetQ_value = Qout_value
            targetQ_value[0, action] = reward + y*max_new_Q

            sess.run(optimizer, 
                    feed_dict={ 
                        input_placeholder:[gen_one_hot(observation)],
                        target_Q: targetQ_value
                        })
            
            if val != "s":
                print(chr(27) + "[2J")
                print(f"{i}/{step}:\t{observation}, {action}, {reward}")
                env.render()
                val = input("Press Enter to step")
            else:
                print(f"{i}/{step}:\t{observation}, {action}, {reward}", end="\r")
     
            total_reward += reward
            observation = new_observation
            if done:
                threshold = 1.0/((i/50) + 10)
                break

        stepList.append(step)
        rewardList.append(total_reward)
print("Percentage of success: " + str(sum(rewardList)/num_episodes))
