import gym
import numpy as np

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

decay = 0.8
discount = 0.95
num_episodes = 2000

for i in range(num_episodes):
    observation = env.reset()
    total_reward = 0
    done = False
    step = 0
    while step < 99:
        step = step + 1
        random_decision = np.random.randn(1, env.action_space.n)
        actions_prob = Q[observation,:] + random_decision*(1.0/(i+1))
        action = np.argmax(actions_prob)

        new_observation, reward, done, _ = env.step(action)
        Q[observation, action] = Q[observation, action] + decay*(reward + discount*np.max(Q[new_observation, :]) - Q[observation, action])

        print(chr(27) + "[2J")
        print(f"{i}/{step}: {observation}, {action}, {reward}")
        env.render()
        input("Press Enter to step")
        total_reward += reward
        observation = new_observation

        if done:
            break
