import gymnasium as gym
from agent import Agent

test_env = gym.make("CarRacing-v2",render_mode="human",continuous=False)
#test_env = gym.make("CartPole-v1",render_mode="human")
obs_space = test_env.observation_space.shape
input_len = 1
for space in obs_space:
    input_len *= space
#input_len = test_env.observation_space.n
agent = Agent(input = input_len ,output=test_env.action_space.n)
agent.load()
state, _ = test_env.reset()
#state_ = agent.state_to_flatten(state)
state_ = agent.preprocess_image(state)
print("#################################")
done=False
while not done:
    action = agent.action_dist(state_).sample().item()
    print("action: "+ str(action))
    test_env.render()
    next_state, reward, done, _, _ = test_env.step(action)
    #next_state_ = agent.state_to_flatten(next_state)
    next_state_ = agent.preprocess_image(next_state)
    state_ = next_state_
