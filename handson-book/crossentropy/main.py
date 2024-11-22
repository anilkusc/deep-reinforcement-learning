import gymnasium as gym
from agent import Agent

env = gym.make("CartPole-v1",render_mode=None)
obs_space = env.observation_space.shape
input_len = 1
for space in obs_space:
    input_len *= space
#input_len = env.observation_space.n
agent = Agent(input = input_len,output=env.action_space.n,learning_rate=0.01)
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
        action = agent.action_selector(state)
        env.render()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated or (move >= max_move)
        total_reward += reward
        agent.replay_memory.append((state,action))
        state = next_state
    all_episodes.append({"return":total_reward,"episodes":agent.replay_memory})
    print(" total reward: "+ str(int(total_reward)))
    agent.replay_memory = []
    j = 0
    if ((i+1) % 100) == 0:
        s,a = agent.filter_best_trajectories_as_tensor(all_episodes)
        loss = agent.train(s,a)
        agent.monitor(total_reward,loss,((i)/100))
        all_episodes = []
agent.save()
agent.writer.close()