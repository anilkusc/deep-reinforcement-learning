import gymnasium as gym
import numpy as np
import ale_py


class CompatibleWithPytorchConv(gym.ObservationWrapper):
    def __init__(self, env):
        super(CompatibleWithPytorchConv, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, obs):
        return np.moveaxis(obs, 2, 0)
        #weights = np.array([0.2989, 0.5870, 0.1140])
        #grayscale_image = np.dot(obs, weights).astype(np.uint8)
        #return grayscale_image
        #normalized = np.round(grayscale_image / 255.0, 3)
        #return normalized.flatten()

def make_env(env_name,render_mode=None):
    gym.register_envs(ale_py)
    env = gym.make(env_name,render_mode=render_mode)
    env = CompatibleWithPytorchConv(env)
    return env