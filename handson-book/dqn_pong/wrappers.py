import gymnasium as gym
import numpy as np
import ale_py


class ConvertToGrayscale(gym.ObservationWrapper):
    def __init__(self, env):
        super(ConvertToGrayscale, self).__init__(env)
        
    def observation(self, obs):
        weights = np.array([0.2989, 0.5870, 0.1140])
        grayscale_image = np.dot(obs, weights).astype(np.uint8)
        normalized = np.round(grayscale_image / 255.0, 3)
        return normalized

def make_env(env_name,render_mode=None):
    gym.register_envs(ale_py)
    env = gym.make(env_name,render_mode=render_mode)
    env = ConvertToGrayscale(env)
    return env