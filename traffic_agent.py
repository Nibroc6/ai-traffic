import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, obs_size, num_lights, num_speeds):
        super(DQN, self).__init__()
        self.num_lights = num_lights
        self.num_speeds = num_speeds
        
        # Shared features
        self.features = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        ).to(device)
        
        # Separate heads for lights and speeds
        self.lights_head = nn.Linear(256, num_lights).to(device)
        self.speeds_head = nn.Linear(256, num_speeds).to(device)

    def forward(self, x):
        features = self.features(x)
        lights = torch.sigmoid(self.lights_head(features))  # Binary outputs
        speeds = self.speeds_head(features)  # Continuous outputs
        return lights, speeds

class TrafficControlAgent:
    def __init__(self, env, memory_size=100000, batch_size=1024):
        self.env = env
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Calculate input size
        self.obs_size = (
            env.observation_space['nodes'].shape[0] * env.observation_space['nodes'].shape[1] +
            env.observation_space['edges'].shape[0] * env.observation_space['edges'].shape[1]
        )
        
        # Get action space dimensions
        self.num_lights = env.action_space['lights'].n
        self.num_speeds = env.action_space['speeds'].shape[0]
        
        # Initialize networks with correct parameters
        self.policy_net = DQN(self.obs_size, self.num_lights, self.num_speeds)
        self.target_net = DQN(self.obs_size, self.num_lights, self.num_speeds)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.001)
        
        # Memory buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10

    def preprocess_observation(self, observation):
        """Convert dict observation to flat array"""
        node_obs = observation['nodes'].flatten()
        edge_obs = observation['edges'].flatten()
        return np.concatenate([node_obs, edge_obs])

    def select_action(self, state):
        if random.random() < self.epsilon:
            return {
                'lights': self.env.action_space['lights'].sample(),
                'speeds': self.env.action_space['speeds'].sample()
            }
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self.preprocess_observation(state)).unsqueeze(0).to(device)
            lights_pred, speeds_pred = self.policy_net(state_tensor)
            
            # Convert predictions to actions
            lights = (lights_pred.cpu().numpy() > 0.5).astype(np.int8)[0]
            speeds = np.clip(
                speeds_pred.cpu().numpy()[0],
                self.env.action_space['speeds'].low[0],
                self.env.action_space['speeds'].high[0]
            )
            
            return {
                'lights': lights,
                'speeds': speeds
            }

    def train(self, num_episodes):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            
            while True:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Store transition in memory
                self.memory.append((
                    self.preprocess_observation(state),
                    action,
                    reward,
                    self.preprocess_observation(next_state),
                    done
                ))
                
                total_reward += reward
                state = next_state
                
                # Train on random batch from memory
                if len(self.memory) > self.batch_size:
                    self._train_step()
                
                if done:
                    break
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print progress with GPU stats
            if torch.cuda.is_available():
                print(f"Episode {episode + 1}, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.2f}, "
                      f"CUDA Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            else:
                print(f"Episode {episode + 1}, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.2f}")

    def _train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.vstack(states)).to(device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Get current predictions
        current_lights, current_speeds = self.policy_net(states)
        
        # Get next state predictions
        with torch.no_grad():
            next_lights, next_speeds = self.target_net(next_states)
        
        # Calculate targets
        lights_target = torch.FloatTensor(np.vstack([action['lights'] for action in actions])).to(device)
        speeds_target = torch.FloatTensor(np.vstack([action['speeds'] for action in actions])).to(device)
        
        # Calculate losses separately for lights and speeds
        lights_loss = nn.BCELoss()(current_lights, lights_target)
        speeds_loss = nn.MSELoss()(current_speeds, speeds_target)
        
        # Combined loss
        loss = lights_loss + speeds_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def save(self, path):
        """Save the model and training state"""
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        """Load the model and training state"""
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']