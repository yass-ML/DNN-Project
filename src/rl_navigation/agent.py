import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class RLAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99  # Discount factor

    def select_action(self, state):
        """
        Select action based on current policy.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        # TODO: Implement sampling logic (e.g., Categorical) for training
        # or max for inference
        action = torch.argmax(probs, dim=1).item()
        return action

    def train(self, env, episodes=100):
        """
        Main training loop for the RL agent.
        """
        print(f"Starting RL training for {episodes} episodes...")
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                # TODO: Store transition in memory buffer
                # TODO: Perform policy update step (PPO, A2C, REINFORCE, etc.)
                
                state = next_state
                total_reward += reward
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")
        
        print("RL Training complete.")
