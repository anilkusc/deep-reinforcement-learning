import torch
import gymnasium as gym

class ConvertStateToTensor(gym.Wrapper):
    def __init__(self, env):
        super(ConvertStateToTensor, self).__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state_tensor = torch.from_numpy(next_state).float().to(self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
        terminated_tensor = torch.tensor(1 if terminated else 0, dtype=torch.float32).to(self.device)
        return next_state_tensor, reward_tensor, terminated_tensor, truncated, info

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        return torch.from_numpy(state).float().to(self.device), info

def make_env(env_name,render_mode=None):
    env = gym.make(env_name,render_mode=render_mode)
    #env = ConvertStateToTensor(env)
    return env