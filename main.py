import gym
import follow_the_leader_gym

env = gym.make("Follow-The-Leader-v0")
obs = env.reset()

for _ in range(500):
    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)
    print(obs, reward)
    env.render()

env.close()
