import gym
# Import environment
env = gym.make('FetchPickAndPlace-v1')
# initialize the environment
env.reset()
# 1000 cycles 
for _ in range(1000):
    # drawing
    # env.render()
    # perform an action
    a = env.action_space.sample()
    s, r, d, _ = env.step(a) # take a random action
    print(s)
# close
env.close()
