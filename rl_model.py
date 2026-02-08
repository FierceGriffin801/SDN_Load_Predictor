import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNClassifier:
    """
    Reinforcement Learning Baseline (DQN) wrapped as a Classifier.
    Treats classification as a Contextual Bandit problem:
    - State: Input Features
    - Action: Predicted Class
    - Reward: 1 if correct, 0 if incorrect
    """
    def __init__(self, input_dim, output_dim, gamma=0.99, lr=0.001, batch_size=64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.q_net = QNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def fit(self, X, y, epochs=10):
        print(f"Training DQN Agent (RL Baseline) for {epochs} epochs...")
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Epsilon Decay Per Step
        # e.g., 0.99995^20000 ~= 0.36
        # We want to reach min epsilon around end of training
        total_steps = len(X) * epochs
        self.epsilon_decay = np.power(self.epsilon_min / self.epsilon, 1 / (total_steps * 0.8))
        
        step_count = 0
        
        # Simulate Episodes
        for epoch in range(epochs):
            total_reward = 0
            
            # Shuffle data for "Experience Replay" feel
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            
            for i in indices:
                state = X_tensor[i].unsqueeze(0)
                target = y_tensor[i].item()
                
                # Epsilon-Greedy Action
                if random.random() < self.epsilon:
                    action = random.randint(0, self.output_dim - 1)
                else:
                    with torch.no_grad():
                        q_values = self.q_net(state)
                        action = torch.argmax(q_values).item()
                
                # Reward Calculation
                reward = 1.0 if action == target else -1.0
                total_reward += reward
                
                # Store Match in Buffer
                self.replay_buffer.append((state, action, reward))
                
                # Train Experience Replay (Every 10 steps to speed up)
                if step_count % 10 == 0 and len(self.replay_buffer) > self.batch_size:
                    self._train_step()
                    
                # Decay Epsilon Per Step
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                step_count += 1
            
            print(f"Epoch {epoch+1}/{epochs} | Avg Reward: {total_reward/len(X):.4f} | Epsilon: {self.epsilon:.4f}")

    def _train_step(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards = zip(*batch)
        
        states = torch.cat(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        
        # Q(s, a)
        q_values = self.q_net(states).gather(1, actions)
        
        # Target: Reward (Contextual Bandit has no future state transition)
        target_q = rewards 
        
        loss = self.criterion(q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, X):
        self.q_net.eval()
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            q_values = self.q_net(X_tensor)
            actions = torch.argmax(q_values, dim=1)
        return actions.numpy()
