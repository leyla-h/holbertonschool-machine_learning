#!/usr/bin/env python3
"""Displays a game played by the agent trained by train.py."""
import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from rl.core import Processor


class GymnasiumWrapper(gym.Wrapper):
    """Wraps a gymnasium environment so it exposes the old gym API
    that keras-rl expects (reset returns only the observation, step
    returns obs, reward, done, info).
    """

    def reset(self, **kwargs):
        """Resets the environment and returns only the observation"""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Steps the environment, merging terminated/truncated into a
        single done flag
        """
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, *args, **kwargs):
        """Renders the environment"""
        return self.env.render()


class AtariProcessor(Processor):
    """Preprocesses Atari frames and rewards for the DQN agent"""

    def process_observation(self, observation):
        """Resizes and converts an observation to grayscale uint8"""
        from PIL import Image
        img = Image.fromarray(observation)
        img = img.resize((84, 84)).convert('L')
        return np.array(img, dtype=np.uint8)

    def process_state_batch(self, batch):
        """Normalizes a batch of stacked frames to floats in [0, 1]"""
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clips rewards to [-1, 1]"""
        return np.clip(reward, -1.0, 1.0)


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = GymnasiumWrapper(env)

    num_actions = env.action_space.n
    window_length = 4

    model = load_model('policy.h5')

    memory = SequentialMemory(limit=1000000, window_length=window_length)
    processor = AtariProcessor()
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=num_actions,
        memory=memory,
        processor=processor,
        policy=policy)

    dqn.compile('adam', metrics=['mae'])

    dqn.test(env, nb_episodes=5, visualize=False)
