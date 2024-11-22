import gymnasium as gym
from agent import Agent

test_env = gym.make("CartPole-v1",render_mode="human")
obs_space = test_env.observation_space.shape
input_len = 1
for space in obs_space:
    input_len *= space
#input_len = test_env.observation_space.n
agent = Agent(input = input_len ,output=test_env.action_space.n)
agent.load()
for name, param in agent.model.state_dict().items():
    print(f"Parameter: {name}\nValue: {param}\n")
state, _ = test_env.reset()
done=False
total_reward = 0
while not done:
    action = agent.action_selector(state)
    print("action: "+ str(action))
    test_env.render()
    next_state, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward
    done = terminated or truncated
    state = next_state
print("total_reward: "+ str(total_reward))
