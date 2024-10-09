import gymnasium as gym
from agent import Agent

env = gym.make("CarRacing-v2",render_mode=None,continuous=False)
obs_space = env.observation_space.shape
input_len = 1
for space in obs_space:
    input_len *= space
#input_len = env.observation_space.n    
agent = Agent(input = input_len,output=env.action_space.n,replay_memory=4096,memory_batch_size=2048)
epochs = 100
max_move = 1028
total_loss = 0
for episode in range(epochs):
    print("Episode: "+str(episode))
    state, _ = env.reset()
    state_ = agent.preprocess_image(state)
    done = False
    move = 0
    total_loss = 0
    total_reward = 0
    agent.transitions = []
    while not done:
        move += 1
        action = agent.Action_Selector(state_)
        #env.render()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated #or (move >= max_move)
        next_state_ = agent.preprocess_image(next_state)
        total_reward += reward
        agent.transitions.append((state_,action,reward))
        state_ = next_state_
        if done:
            print("done")

    loss = agent.REINFORCE()
    print("total move: "+str(move) + " total_reward: "+ str(total_reward) + " epsilon : " + str(agent.epsilon))
agent.save()