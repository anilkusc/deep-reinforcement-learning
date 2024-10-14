import gymnasium as gym
from agent import Agent
import numpy as np
#env = gym.make("CarRacing-v2",render_mode=None,continuous=False)
env = gym.make("CartPole-v1",render_mode=None)
obs_space = env.observation_space.shape
input_len = 1
for space in obs_space:
    input_len *= space
#input_len = env.observation_space.n
agent = Agent(input = input_len,output=env.action_space.n,learning_rate=0.0005) #000005
epochs = 1000
max_move = 9999
reward_max = 0
for episode in range(epochs):
    print("Episode: "+str(episode+1) + "/"+str(epochs))
    state, _ = env.reset()
    #state_ = agent.preprocess_image(state)
    state_ = agent.state_to_flatten(state)
    done = False
    move = 0
    total_loss_actor = 0
    total_loss_critic = 0
    total_reward = 0
    trajectories = []
    while not done:
        move += 1
        dist = agent.action_dist(state_)
        action = dist.sample()
        env.render()
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        #reward *= 100
        done = terminated or truncated or (move >= max_move)
        #next_state_ = agent.preprocess_image(next_state)
        next_state_ = agent.state_to_flatten(next_state)
        total_reward += reward
        #trajectories.append((reward,state_,action,next_state_,done))
        actor_loss,critic_loss = agent.update_actor_critic(reward,state_,dist,action,next_state_,done)
        total_loss_actor += actor_loss
        total_loss_critic += critic_loss
        state_ = next_state_
    #total_loss_actor,total_loss_critic = agent.update_actor_critic_trajectory(trajectories)
    if total_reward >= reward_max:
        agent.save()
        reward_max = total_reward
    print("total move: "+str(move) + " total reward: "+ str(int(total_reward)) + " actor loss: " + str(total_loss_actor.item())+ " critic loss: " + str(total_loss_critic.item()))
#agent.save()