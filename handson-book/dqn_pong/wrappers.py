import gymnasium as gym
import numpy as np
import ale_py
from collections import deque
import cv2
import torch
import collections

class CompatibleWithPytorchConv(gym.ObservationWrapper):
    def __init__(self, env):
        super(CompatibleWithPytorchConv, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, obs):
        return np.moveaxis(obs, 2, 0)

class NormalizeImageData(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizeImageData, self).__init__(env)

    def observation(self, obs):
        return np.round(obs / 255.0, 3)


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

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(2)

    def step(self, action):
        action = 2 + action
        return self.env.step(action)

class RewardHandler(gym.Wrapper):
    def __init__(self, env):
        super(RewardHandler, self).__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward == 0:
            reward = -0.1
        if reward == 1:
            reward = 1.1
        if reward == -1:
            reward = -1.1
        return obs, reward, terminated, truncated, info

class ConvertStateToTensor(gym.Wrapper):
    def __init__(self, env):
        super(ConvertStateToTensor, self).__init__(env)

    def step(self, action,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state_tensor = torch.from_numpy(next_state).float().to(device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
        terminated_tensor = torch.tensor(1 if terminated else 0, dtype=torch.float32).to(device)
        return next_state_tensor, reward_tensor, terminated_tensor, truncated, info

class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=3):
        super(FrameSkip, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        
        return self.env.reset(**kwargs)


def make_env(env_name,render_mode=None):
    gym.register_envs(ale_py)
    env = gym.make(env_name,render_mode=render_mode)
    env = FrameSkip(env) 
    env = ProcessFrame84(env)
    env = FrameStackPyTorch(env)
    env = CompatibleWithPytorchConv(env)
    env = NormalizeImageData(env)
    #env = ActionWrapper(env)
    #env = RewardHandler(env)
    #env = ConvertStateToTensor(env)
    return env