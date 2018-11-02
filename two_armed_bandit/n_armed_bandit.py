import tensorflow as tf
import numpy as np

hidden_bandits = [0.2, 0, -0.2, -5]
num_bandits = len(hidden_bandits)
def pullBandit(bandit):
    result = np.random.randn(1)

    if result > bandit: # if randomly a value is greater than bandit, that's 1
        return 1
    else:
        return -1


learning_rate = 0.001
tf.reset_default_graph()
W = tf.Variable(tf.ones([num_bandits])) # 4x1
chosen_action = tf.argmax(W, 0)

reward_holder = tf.placeholder(tf.float32, (1))
action_holder = tf.placeholder(tf.int32, (1))
responsible_weight = tf.slice(W, action_holder, [1])
loss = -(tf.log(responsible_weight)*reward_holder) # -log(π)*A
update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


e = 0.3 #probability of taking random action
total_episodes = 1000
total_rewards = np.zeros(num_bandits)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(total_episodes):

        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits) # take random action
        else:
            action = sess.run(chosen_action) # take best action

        reward = pullBandit(hidden_bandits[action])
        _, resp, weights_value = sess.run([update, responsible_weight, W],
                                          feed_dict = {
                                              reward_holder: [reward],
                                              action_holder: [action]
                                          })

        total_rewards[action] += reward

        if i%50 == 0:
            print("Running reward for the " + str(num_bandits) + " bandits: " + str(total_rewards))
        i+=1

print("The agent thinks bandit " + str(np.argmax(weights_value)+1) + " is the most promising....")
if np.argmax(weights_value) == np.argmax(-np.array(hidden_bandits)):
        print("...and it was right!")
else:
        print("...and it was wrong!")
