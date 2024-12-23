from agent import Agent
from wrappers import make_env

env = make_env("ALE/Pong-v5",render_mode="rgb_array")

agent = Agent(input = env.observation_space.shape,output=env.action_space.n,learning_rate=0.001,tensorboard=True,replay_memory=25000,memory_batch_size=2048,exploration_decay_rate=0.991,hidden=256,min_epsilon=0.0001,conv_hidden=16)

print(agent.conv)
print(agent.model)
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
        agent.replay_memory.append((reward,state,action,next_state,done))
        state = next_state
        if len(agent.replay_memory) > agent.memory_batch_size and (move % 50 == 0):
            loss = agent.train()
            total_loss += float(loss)
    agent.epsilon_decay()
    agent.save()
    print("Total Loss: "+str(total_loss) + " Total Reward: "+str(total_reward)+ " Epsilon: "+str(agent.epsilon))    
    if agent.tensorboard:
        agent.writer.add_scalar("total_reward", total_reward, episode)
        agent.writer.add_scalar("epsilon", agent.epsilon, episode)
        agent.writer.add_scalar("total_move", move, episode)
        agent.writer.add_scalar("total_loss", total_loss, episode)