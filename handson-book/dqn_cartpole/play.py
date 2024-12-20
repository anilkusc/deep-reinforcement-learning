from agent import Agent
from wrappers import make_env
env = make_env('CartPole-v1',render_mode="human")
input_len = 1
for space in env.observation_space.shape:
    input_len *= space
agent = Agent(input = input_len ,output=env.action_space.n,hidden=128,tensorboard=False)
agent.load()
agent.epsilon = 0
for name, param in agent.model.state_dict().items():
    print(f"Parameter: {name}\nValue: {param}\n")
state, _ = env.reset(seed=123)
done=False
total_reward = 0
while not done:
    action = agent.action_selector(state)
    print("action: "+ str(action))
    env.render()
    next_state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated
    state = next_state
print("total_reward: "+ str(total_reward))