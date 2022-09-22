import gym
import follow_the_leader_gym

env = gym.make("Follow-The-Leader-v0")
env.reset()

for _ in range(1000):
    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)
    env.render()

env.close()
