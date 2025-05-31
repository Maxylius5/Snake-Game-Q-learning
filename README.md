# 🐍 Snake AI – Q-Learning Reinforcement Learning Agent

This repository contains a Python implementation of the classic Snake game where the agent learns to play using **Q-learning**, a fundamental reinforcement learning algorithm.

The goal of this project is to apply AI techniques in a controlled environment and explore core concepts such as state space design, reward engineering, and training stability.

---

## 🚀 Features

- Reinforcement Learning using **Q-learning with ε-greedy policy**
- Snake game built with **Pygame**
- **Compact state representation** for efficient learning
- Experience replay with batch training
- Dynamic plot of performance and loss
- Easily switch between **training** and **inference** modes
- Save/load trained models



## 🧠 How It Works

The AI uses a 11 element input state that encodes:
- Danger in front, left, right
- Relative direction of food
- Current movement direction

**Q-learning** is used to learn the optimal action-value function:
- Actions: `[straight, right turn, left turn]`
- Reward design:
  - +10 for eating food
  - -10 for hitting a wall or itself
  - Small negative reward for each time step to encourage efficiency



