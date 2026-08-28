README

Deep Q-Learning: Playing Atari's Breakout

This project trains a reinforcement learning agent to play Atari's Breakout using Deep Q-Learning, built with keras-rl2, gymnasium, and keras.

Files

train.py — trains a DQNAgent on ALE/Breakout-v5 using an epsilon-greedy policy, a convolutional Q-network, and experience replay via SequentialMemory. Saves the trained network to policy.h5.
play.py — loads policy.h5 and lets the trained agent play Breakout using a greedy policy, rendering the game so you can watch it play.

Requirements

gymnasium[atari]
ale-py (and AutoROM to install the Atari ROMs)
keras-rl2
tensorflow
Pillow
numpy

Usage

Train the agent: ./train.py
Watch the trained agent play: ./play.py
