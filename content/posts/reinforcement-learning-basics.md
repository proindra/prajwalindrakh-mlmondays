---
title: "Reinforcement Learning Basics: Training Agents to Make Decisions"
excerpt: "Learn the fundamentals of reinforcement learning, from basic concepts to implementing your first RL agent using modern frameworks."
author: "Dr. Kevin Park"
date: "2024-11-10"
tags: ["reinforcement-learning", "q-learning", "policy-gradient", "gym"]
image: "/images.jpg"
---

# Reinforcement Learning Basics: Training Agents to Make Decisions

Reinforcement Learning (RL) represents one of the most exciting frontiers in AI, enabling agents to learn optimal behaviors through interaction with their environment. Unlike supervised learning, RL agents learn from rewards and punishments, much like how humans learn.

## What is Reinforcement Learning?

RL is a machine learning paradigm where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. The key components are:

- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State**: Current situation of the agent
- **Action**: What the agent can do
- **Reward**: Feedback from the environment
- **Policy**: Strategy for choosing actions

## The RL Framework

### Markov Decision Process (MDP)

```python
import numpy as np
from typing import Tuple, List, Dict

class SimpleMDP:
    def __init__(self, states: List[str], actions: List[str]):
        self.states = states
        self.actions = actions
        self.n_states = len(states)
        self.n_actions = len(actions)
        
        # Transition probabilities P(s'|s,a)
        self.transition_probs = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        # Reward function R(s,a,s')
        self.rewards = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        # Discount factor
        self.gamma = 0.9
    
    def set_transition(self, state: int, action: int, next_state: int, prob: float):
        """Set transition probability"""
        self.transition_probs[state, action, next_state] = prob
    
    def set_reward(self, state: int, action: int, next_state: int, reward: float):
        """Set reward for transition"""
        self.rewards[state, action, next_state] = reward
    
    def get_expected_reward(self, state: int, action: int) -> float:
        """Calculate expected reward for state-action pair"""
        return np.sum(self.transition_probs[state, action] * 
                     np.sum(self.rewards[state, action], axis=1))
```

## Q-Learning: The Foundation

Q-Learning learns the quality (Q-value) of state-action pairs without needing a model of the environment.

```python
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, 
                 actions: List[str],
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1):
        
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-table: Q(state, action) -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def get_action(self, state: str) -> str:
        """Epsilon-greedy action selection"""
        
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(self.actions)
        else:
            # Exploitation: best known action
            q_values = [self.q_table[state][action] for action in self.actions]
            max_q = max(q_values)
            
            # Handle ties by random selection
            best_actions = [action for action, q_val in 
                          zip(self.actions, q_values) if q_val == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state: str, action: str, reward: float, 
                      next_state: str, done: bool):
        """Update Q-value using Q-learning update rule"""
        
        current_q = self.q_table[state][action]
        
        if done:
            # Terminal state
            target_q = reward
        else:
            # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            next_q_values = [self.q_table[next_state][a] for a in self.actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self, decay_rate: float = 0.995):
        """Decay exploration rate over time"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)
```

## Implementing a Simple Environment

```python
import gym
from gym import spaces

class GridWorldEnv(gym.Env):
    """Simple grid world environment"""
    
    def __init__(self, size: int = 5):
        super(GridWorldEnv, self).__init__()
        
        self.size = size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=size-1, 
                                          shape=(2,), dtype=np.int32)
        
        # Define actions
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        # Goal position
        self.goal = (size-1, size-1)
        
        # Reset environment
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.agent_pos = (0, 0)
        return np.array(self.agent_pos)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take action and return next state, reward, done, info"""
        
        # Calculate new position
        dx, dy = self.actions[action]
        new_x = max(0, min(self.size-1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size-1, self.agent_pos[1] + dy))
        
        self.agent_pos = (new_x, new_y)
        
        # Calculate reward
        if self.agent_pos == self.goal:
            reward = 100  # Goal reached
            done = True
        else:
            reward = -1   # Step penalty
            done = False
        
        return np.array(self.agent_pos), reward, done, {}
    
    def render(self):
        """Visualize current state"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        grid[self.goal[0]][self.goal[1]] = 'G'
        
        for row in grid:
            print(' '.join(row))
        print()
```

## Training Loop

```python
def train_q_learning(env, agent, episodes: int = 1000):
    """Train Q-learning agent"""
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            # Convert state to string for Q-table
            state_str = f"{state[0]},{state[1]}"
            
            # Choose action
            action = agent.get_action(state_str)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state_str = f"{next_state[0]},{next_state[1]}"
            
            # Update Q-value
            agent.update_q_value(state_str, action, reward, next_state_str, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards

# Usage
env = GridWorldEnv(size=5)
agent = QLearningAgent(actions=[0, 1, 2, 3])
rewards = train_q_learning(env, agent, episodes=1000)
```

## Deep Q-Networks (DQN)

For complex environments, we use neural networks to approximate Q-values:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks
        self.q_network = DQN(state_size, 64, action_size)
        self.target_network = DQN(state_size, 64, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
```

## Policy Gradient Methods

Instead of learning Q-values, policy gradient methods directly learn the policy:

```python
class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCEAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.01):
        self.policy_network = PolicyNetwork(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """Select action based on policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_network(state_tensor)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        # Save log probability for training
        self.saved_log_probs.append(action_dist.log_prob(action))
        
        return action.item()
    
    def update_policy(self, gamma: float = 0.99):
        """Update policy using REINFORCE algorithm"""
        
        # Calculate discounted rewards
        discounted_rewards = []
        running_reward = 0
        
        for reward in reversed(self.rewards):
            running_reward = reward + gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        
        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
                           (discounted_rewards.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.saved_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.saved_log_probs.clear()
        self.rewards.clear()
```

## Practical Applications

### 1. Game Playing

```python
# Using OpenAI Gym for Atari games
import gym

def train_atari_agent():
    env = gym.make('Breakout-v4')
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )
    
    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.replay()
        
        if episode % 100 == 0:
            agent.update_target_network()
```

### 2. Robotics Control

```python
def train_robot_arm():
    """Example for continuous control tasks"""
    
    # Use environments like MuJoCo or PyBullet
    env = gym.make('Reacher-v2')
    
    # For continuous actions, use algorithms like:
    # - Deep Deterministic Policy Gradient (DDPG)
    # - Proximal Policy Optimization (PPO)
    # - Soft Actor-Critic (SAC)
    
    pass
```

## Best Practices

1. **Start Simple**: Begin with tabular methods before moving to deep RL
2. **Environment Design**: Reward shaping is crucial for learning
3. **Exploration vs Exploitation**: Balance is key for effective learning
4. **Hyperparameter Tuning**: Learning rate, discount factor, and exploration rate matter
5. **Stability**: Use techniques like target networks and experience replay
6. **Evaluation**: Test on multiple random seeds and environments

## Common Challenges

- **Sample Efficiency**: RL often requires many interactions
- **Stability**: Training can be unstable and sensitive to hyperparameters
- **Reward Design**: Poorly designed rewards can lead to unexpected behaviors
- **Generalization**: Agents may overfit to specific environments

Reinforcement learning opens up possibilities for creating truly autonomous agents that can adapt and improve their decision-making over time. While challenging, the potential applications are limitless.

---

*Next week: Advanced RL topics including multi-agent systems and hierarchical reinforcement learning*