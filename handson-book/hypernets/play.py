from agent import Agent
from wrappers import make_env

env = make_env("CarRacing-v2", render_mode="human")
agent = Agent(
    input=env.observation_space.shape,
    output=env.action_space.n,
    hidden=512,
    min_epsilon=0.000001,
    conv_hidden=16,
    max_epsilon=0.1
)
agent.load(model_name="model_ac.pth")
print(agent.model)

state, _ = env.reset(seed=123)
done = False
total_reward = 0
move = 0

while not done:
    move += 1
    action = agent.action_selector(state)
    print("action: " + str(action))
    env.render()
    
    next_state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    
    done = terminated or truncated
    state = next_state
    
    print("reward: " + str(reward))
print("total_reward: " + str(total_reward))

env.close() 