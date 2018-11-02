import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np

class ContextBandit():
    state = 0

    def __init__(self):
        self.hidden_bandits = np.array([
            [0.2, 0, -0.0, -5], # state 0, where action 4 is best
            [0.1, -5, 1, 0.25], # state 1, where action 2 is best
            [-5, 5, 5, 5]       # state 2, where action 1 is best
        ]) # different bandits is determined by the state

        self.num_bandits = self.hidden_bandits.shape[0]
        self.num_actions = self.hidden_bandits.shape[1]

    def getState(self):
        self.state = np.random.randint(0, self.num_bandits)
        return self.state

    def pullArm(self, action):
        bandit = self.hidden_bandits[self.state, action]
        result = np.random.randn(1)

        if result > bandit: # if randomly a value is greater than bandit, that's 1
            return 1
        else:
            return -1

class Agent():

    def __init__(self, learning_rate, no_of_states, fully_connected_size):
        self.state_placeholder = tf.placeholder(tf.int32, (1))
        state_one_hot = slim.one_hot_encoding(self.state_placeholder,
                                              no_of_states)
        output = slim.fully_connected(state_one_hot, fully_connected_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, axis=0)

        self.reward_holder = tf.placeholder(tf.float32, (1))
        self.action_holder = tf.placeholder(tf.int32, (1))
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        self.update = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)\
            .minimize(self.loss)

tf.reset_default_graph()
cBandit = ContextBandit()
agent = Agent(learning_rate=0.001,
                no_of_states=cBandit.num_bandits,
                fully_connected_size=cBandit.num_actions)

e = 0.1 #probability of taking random action
total_episodes = 10000
total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(total_episodes):
        current_state = cBandit.getState()

        if np.random.rand() < e:
            action = np.random.randint(cBandit.num_actions)
        else:
            action = sess.run(agent.chosen_action,
                              feed_dict={
                                  agent.state_placeholder: [current_state]
                              })

        reward = cBandit.pullArm(action)

        sess.run(agent.update,
                 feed_dict={
                     agent.reward_holder: [reward],
                     agent.action_holder: [action],
                     agent.state_placeholder: [current_state]
                 })

        total_reward[current_state, action] += reward

        if i%500 == 0:
            print("Mean reward for each of the " +
                  str(cBandit.num_bandits) + "bandits: " +
                  str(np.mean(total_reward,axis=1)))

print(cBandit.hidden_bandits)
print(total_reward)
