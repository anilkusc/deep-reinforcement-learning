import torch
import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class CompatibleWithPytorchConvOld(gym.ObservationWrapper):
    def __init__(self, env):
        super(CompatibleWithPytorchConvOld, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (1, old_shape[0], old_shape[1])  # Gri tonlamada tek kanal olacağı için 1 kanal
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, obs):
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Gri tonlamalı görüntüyü (H, W) boyutunda yaparken (1, H, W) boyutuna getirmek için reshape yapma
        gray_obs = np.expand_dims(gray_obs, axis=0)
        gray_obs = gray_obs / 255.0
        return gray_obs

class CompatibleWithPytorchConv2(gym.ObservationWrapper):
    def __init__(self, env):
        super(CompatibleWithPytorchConv2, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, obs):
        obs = obs / 255.0
        return np.moveaxis(obs, 2, 0)

class CompatibleWithPytorchConv(gym.ObservationWrapper):
    def __init__(self, env):
        super(CompatibleWithPytorchConv, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (1, old_shape[0], old_shape[1])  # Tek kanala indir
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, obs):
        # RGB'yi grayscale'e çevir
        grayscale_obs = np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Normalize et ve boyut ekle
        grayscale_obs = grayscale_obs / 255.0
        grayscale_obs = grayscale_obs[np.newaxis, :, :]  # Boyut ekle
        
        return grayscale_obs
    
class FlattenNormalizeAndGrayscaleImageData(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlattenNormalizeAndGrayscaleImageData, self).__init__(env)
        
        obs_shape = self.observation_space.shape
        if len(obs_shape) == 3:
            height, width, _ = obs_shape
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(height * width,), dtype=np.float32
            )

    def observation(self, obs):
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) if len(obs.shape) == 3 else obs
        return (gray_obs / 255.0).astype(np.float32).flatten()

class NormalizeAndPermuteObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizeAndPermuteObservation, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1, 96, 96), dtype=np.float32  # Kanalı 1 olarak ayarlıyoruz
        )

    def observation(self, obs):
        obs = obs / 255.0  # Normalize
        if len(obs.shape) == 3:  # Eğer RGB ise
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # Grayscale'e çevir
        return np.expand_dims(obs, axis=0)  # (H, W) -> (1, H, W)
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=3):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info

class FrameStackPyTorch(gym.Wrapper):
    def __init__(self, env, num_frames=3):
        super(FrameStackPyTorch, self).__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        obs_space = env.observation_space
        new_shape = (num_frames * obs_space.shape[0], obs_space.shape[1], obs_space.shape[2])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_frames):
            self.frames.append(np.zeros_like(obs))
        self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        return np.concatenate(list(self.frames), axis=0)

def make_env(env_name, render_mode=None, continuous=False):
    env = gym.make(env_name, render_mode=render_mode, continuous=continuous)
    env = CompatibleWithPytorchConv(env)
    env = SkipFrame(env)
    env = FrameStackPyTorch(env)
    return env