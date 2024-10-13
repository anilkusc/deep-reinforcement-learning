import gymnasium as gym
from agent import Agent

env = gym.make("CarRacing-v2",render_mode=None,continuous=False)
obs_space = env.observation_space.shape
input_len = 1
for space in obs_space:
    input_len *= space
#input_len = env.observation_space.n    
agent = Agent(input = input_len,output=env.action_space.n,learning_rate=0.00001) #000005
epochs = 1000
max_move = 1028
total_loss = 0
for episode in range(epochs):
    print("Episode: "+str(episode) + "/"+str(epochs))
    state, _ = env.reset()
    state_ = agent.preprocess_image(state)
    done = False
    move = 0
    total_loss_actor = 0
    total_loss_critic = 0
    total_reward = 0
    while not done:
        move += 1
        action = agent.Action_Selector(state_)
        env.render()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated #or (move >= max_move)
        next_state_ = agent.preprocess_image(next_state)
        total_reward += reward
        actor_loss,critic_loss = agent.update_actor_critic(reward,state_,action,next_state_,done)
        total_loss_actor += actor_loss
        total_loss_critic += critic_loss
        state_ = next_state_

    print("total move: "+str(move) + " total reward: "+ str(int(total_reward)) + " actor loss : " + str(int(total_loss_actor.item()))+ " critic loss : " + str(int(total_loss_critic.item())))
agent.save()