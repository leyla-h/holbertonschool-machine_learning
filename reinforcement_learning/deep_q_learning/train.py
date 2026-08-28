#!/usr/bin/env python3
"""Trains an agent to play Atari's Breakout using keras-rl2's DQNAgent.
"""
import gymnasium as gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, Permute,
                                      Activation)
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
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
        """Clips rewards to [-1, 1] for training stability"""
        return np.clip(reward, -1.0, 1.0)


def build_model(window_length, num_actions):
    """Builds the convolutional Q-network used by the DQN agent

    Args:
        window_length: number of stacked frames given as input
        num_actions: size of the environment's discrete action space

    Returns:
        a compiled keras Sequential model
    """
    input_shape = (window_length, 84, 84)

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    model.add(Activation('linear'))

    return model


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5')
    env = GymnasiumWrapper(env)

    num_actions = env.action_space.n
    window_length = 4

    model = build_model(window_length, num_actions)

    memory = SequentialMemory(limit=1000000, window_length=window_length)
    processor = AtariProcessor()
    policy = EpsGreedyQPolicy(eps=0.1)

    dqn = DQNAgent(
        model=model,
        nb_actions=num_actions,
        memory=memory,
        processor=processor,
        policy=policy,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0)

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    dqn.fit(env, nb_steps=1750000, log_interval=10000, visualize=False,
            verbose=2)

    dqn.model.save('policy.h5')
