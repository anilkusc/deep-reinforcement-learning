#from agent import Agent
from wrappers import make_env

env = make_env("ALE/Pong-v5",render_mode="human")
obs_space = env.observation_space.shape
input_len = 1
for space in obs_space:
    input_len *= space
#input_len = env.observation_space.n
#agent = Agent(input = input_len,output=env.action_space.n,learning_rate=0.01)
epochs = 5000
max_move = 9999
reward_max = 0
all_episodes = []
for i,episode in enumerate(range(epochs)):
    print("Episode: "+str(episode+1) + "/"+str(epochs))
    state, _ = env.reset()
    done = False
    move = 0
    total_reward = 0
    while not done:
        move += 1
        action = 2
        env.render()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state