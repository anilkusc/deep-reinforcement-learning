from agent import Agent
from wrappers import make_env

env = make_env("ALE/Pong-v5",render_mode="rgb_array")
agent = Agent(input = env.observation_space.shape,output=env.action_space.n,learning_rate=0.01,tensorboard=True)
epochs = 5000
all_episodes = []
for episode in range(epochs):
    print("Episode: "+str(episode+1) + "/"+str(epochs))
    state, _ = env.reset()
    done = False
    move = 0
    while not done:
        move += 1
        action = agent.action_selector(state)
        env.render()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_memory.append((reward,state,action, next_state,done))
        state = next_state
    print("Total move: "+str(move))
    agent.update()