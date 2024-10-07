import gymnasium as gym
from agent import Agent

env = gym.make("CarRacing-v2",render_mode=None,continuous=False)
#env = gym.make('CliffWalking-v0',render_mode=None)
obs_space = env.observation_space.shape
input_len = 1
for space in obs_space:
    input_len *= space
#input_len = env.observation_space.n    
agent = Agent(input = input_len,output=env.action_space.n,replay_memory=4096,memory_batch_size=2048,gamma=0.5,learning_rate=0.00001)
epochs = 1000
max_move = 1028
total_loss = 0
for episode in range(epochs):
    print("Episode: "+str(episode))
    state, _ = env.reset()
    #state_ = agent.state_to_tensor_input(state)
    state_ = agent.preprocess_image(state)
    done = False
    move = 0
    total_loss = 0
    total_reward = 0
    while not done:
        move += 1
        qval = agent.model(state_)
        action = agent.QPolicy(qval)
        env.render()
        next_state, reward, terminated, truncated, _ = env.step(action)
        #if terminated or truncated:
        #    reward = 10000
        #if (reward == -100):
        #    #reward = -1
        #    terminated = True
        done = terminated or truncated #or (move >= max_move)
        
        #next_state_ = agent.state_to_tensor_input(next_state)
        next_state_ = agent.preprocess_image(next_state)
        total_reward += reward
        #Y = agent.calculate_reward(reward,next_state_.unsqueeze(0),(1 if done else 0))
        #X = qval[action].squeeze()
        #loss = agent.train_backpropagation(X,Y)
        #total_loss += loss.item()

        agent.replay_memory.append((state_,action,reward,next_state_,(1 if done else 0)))
        state_ = next_state_
        if done:
            print("done")

    loss = agent.update()
    if agent.epsilon > 0.1:
        agent.epsilon -= (1/epochs)
    print("total move: "+str(move) + " total_reward: "+ str(total_reward) + " epsilon : " + str(agent.epsilon))
    #print("total loss : "+str(total_loss))
agent.save()