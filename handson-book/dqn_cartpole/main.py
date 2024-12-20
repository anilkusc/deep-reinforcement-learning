from agent import Agent
from wrappers import make_env

env = make_env('CartPole-v1')
input_len = 1
for space in env.observation_space.shape:
    input_len *= space

agent = Agent(input =input_len, output=env.action_space.n,learning_rate=0.001,tensorboard=True,replay_memory=10000,memory_batch_size=1024,exploration_decay_rate=0.995,hidden=128,min_epsilon=0.000001)
epochs = 1000
all_episodes = []
for episode in range(epochs):
    print("Episode: "+str(episode+1) + "/"+str(epochs))
    state, _ = env.reset(seed=123)
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
        if len(agent.replay_memory) > agent.memory_batch_size:
            loss = agent.train()
            total_loss += float(loss)
    agent.epsilon_decay()
    agent.target_model.load_state_dict(agent.model.state_dict())
    agent.save()
    #print("last action:"+ str(action))
    print("Total Loss: "+str(total_loss) + " Total Reward: "+str(total_reward)+ " Total Move:" + str(move)+" Epsilon: "+str(agent.epsilon))    
    if agent.tensorboard:
        agent.writer.add_scalar("total_reward", total_reward, episode)
        agent.writer.add_scalar("epsilon", agent.epsilon, episode)
        agent.writer.add_scalar("total_move", move, episode)
        agent.writer.add_scalar("total_loss", total_loss, episode)