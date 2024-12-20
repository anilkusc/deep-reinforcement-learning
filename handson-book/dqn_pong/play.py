from agent import Agent
from wrappers import make_env

env = make_env("ALE/Pong-v5",render_mode="human")
agent = Agent(input = env.observation_space.shape,output=env.action_space.n,hidden=512)
agent.load()
state, _ = env.reset()
done=False
while not done:
    action = agent.action_selector(state)
    env.render()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state