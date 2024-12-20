from agent import Agent
import gymnasium as gym


env = gym.make('CliffWalking-v0',render_mode=None)
o = env.observation_space.n
a = env.action_space.n
agent = Agent(input = o,output=a,learning_rate=0.001,tensorboard=True,replay_memory=1000,memory_batch_size=32,exploration_decay_rate=0.99,hidden=512)
epochs = 1000
all_episodes = []
for episode in range(epochs):
    print("Episode: "+str(episode+1) + "/"+str(epochs))
    state, _ = env.reset()
    done = False
    move = 0
    total_reward = 0
    total_loss = 0.0
    while not done:
        move += 1
        action = agent.action_selector(state)
        env.render()
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        agent.replay_memory.append((reward,state,action, next_state,done))
        state = next_state
        if len(agent.replay_memory)> agent.memory_batch_size:
            loss = agent.update()
            total_loss += float(loss)
    agent.epsilon_decay()
    agent.target_model.load_state_dict(agent.model.state_dict())
    agent.save()
    print("last action:"+ str(action))
    print("Total Loss: "+str(total_loss) + " Total Reward: "+str(total_reward)+ " Epsilon: "+str(agent.epsilon))    
    if agent.tensorboard:
        agent.writer.add_scalar("total_reward", total_reward, episode)
        agent.writer.add_scalar("epsilon", agent.epsilon, episode)
        agent.writer.add_scalar("total_move", move, episode)
        agent.writer.add_scalar("total_loss", total_loss, episode)