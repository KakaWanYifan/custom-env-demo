from stock_env import StockEnv

env = StockEnv()
env.reset()
for i in range(1000):
    env.render()
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(observation)
    print(reward)
    if terminated or truncated:
        break
env.close()