import gymnasium as gym
from agent import Agent

env = gym.make("CarRacing-v2",render_mode=None,continuous=False)
obs_space = env.observation_space.shape
input_len = 1
for space in obs_space:
    input_len *= space
agent = Agent(input = input_len ,output=env.action_space.n,replay_memory=10000,memory_batch_size=2500)
epochs = 100
max_move = 1000
for episode in range(epochs):
    print("Episode: "+str(episode))
    state, _ = env.reset()
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
        next_state, reward, done, _, _ = env.step(action)
        next_state_ = agent.preprocess_image(next_state)
        total_reward += reward
        agent.append_replay_memory(state_,action,reward,next_state_,done)
        state_ = next_state_
        loss = agent.update()
        if done:
            print("done")

    if agent.epsilon > 0.1:
        agent.epsilon -= (1/epochs)
    print("total move: "+str(move) + " total_reward: "+ str(total_reward) + " epsilon : " + str(agent.epsilon))
agent.save()