import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch.nn.parallel import DataParallel
import torch.multiprocessing as mp

# Force CUDA device selection and verification
if torch.cuda.is_available():
    torch.cuda.init()
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

class DQN(nn.Module):
    def __init__(self, obs_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, 1024),  # Much larger network
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(device)

    def forward(self, x):
        return self.network(x)

class TrafficControlAgent:
    def __init__(self, env, memory_size=100000, batch_size=1024):  # Much larger batch size
        self.env = env
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Calculate input/output sizes
        self.obs_size = (
            env.observation_space['nodes'].shape[0] * env.observation_space['nodes'].shape[1] +
            env.observation_space['edges'].shape[0] * env.observation_space['edges'].shape[1]
        )
        
        self.num_lights = env.action_space['lights'].n
        self.num_speeds = env.action_space['speeds'].shape[0]
        self.action_size = self.num_lights + self.num_speeds
        
        # Initialize networks with DataParallel if multiple GPUs are available
        self.policy_net = DQN(self.obs_size, self.action_size)
        self.target_net = DQN(self.obs_size, self.action_size)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.policy_net = DataParallel(self.policy_net)
            self.target_net = DataParallel(self.target_net)
        
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Use a larger learning rate and different optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5)
        
        # Use a more efficient memory storage
        self.memory = deque(maxlen=memory_size)
        
        # Training parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        
        # Parallel environment processing
        self.num_parallel_envs = 4

    def preprocess_observation(self, observation):
        """Convert dict observation to flat array"""
        node_obs = observation['nodes'].flatten()
        edge_obs = observation['edges'].flatten()
        return np.concatenate([node_obs, edge_obs])
    
    def postprocess_action(self, action_tensor):
        """Convert network output to action dict"""
        # Ensure we're working with CPU numpy arrays
        if isinstance(action_tensor, torch.Tensor):
            action_array = action_tensor.detach().cpu().numpy()
        else:
            action_array = action_tensor
            
        # Split into lights and speeds
        light_values = action_array[:self.num_lights]
        speed_values = action_array[self.num_lights:]
        
        # Process lights (threshold at 0)
        lights = (light_values > 0).astype(np.int8)
        
        # Process speeds (clip to valid range)
        speeds = np.clip(
            speed_values,
            self.env.action_space['speeds'].low[0],
            self.env.action_space['speeds'].high[0]
        )
        
        return {
            'lights': lights,
            'speeds': speeds
        }

    def select_action(self, state):
        if random.random() < self.epsilon:
            # Random action
            return {
                'lights': self.env.action_space['lights'].sample(),
                'speeds': self.env.action_space['speeds'].sample()
            }
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self.preprocess_observation(state)).unsqueeze(0).to(device)
            action_values = self.policy_net(state_tensor)
            return self.postprocess_action(action_values[0])
        
    def process_parallel_steps(self, states):
        """Process multiple environment steps in parallel"""
        actions = []
        with torch.no_grad():
            state_batch = torch.FloatTensor(np.vstack([
                self.preprocess_observation(state) for state in states
            ])).to(device)
            action_values = self.policy_net(state_batch)
            for action_value in action_values:
                actions.append(self.postprocess_action(action_value))
        return actions

    def train(self, num_episodes):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            states = [self.env.reset()[0] for _ in range(self.num_parallel_envs)]
            total_rewards = [0] * self.num_parallel_envs
            
            while True:
                # Process multiple environments in parallel
                actions = self.process_parallel_steps(states)
                
                # Step all environments
                next_states = []
                rewards = []
                dones = []
                
                for i, action in enumerate(actions):
                    next_state, reward, done, _, _ = self.env.step(action)
                    next_states.append(next_state)
                    rewards.append(reward)
                    dones.append(done)
                    total_rewards[i] += reward
                    
                    # Store transitions in memory
                    self.memory.append((
                        self.preprocess_observation(states[i]),
                        action,
                        reward,
                        self.preprocess_observation(next_state),
                        done
                    ))
                
                # Train on multiple batches
                if len(self.memory) > self.batch_size:
                    for _ in range(4):  # Multiple training steps per environment step
                        self._train_step()
                
                states = next_states
                
                if any(dones):
                    break
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Calculate average reward across parallel environments
            avg_reward = sum(total_rewards) / self.num_parallel_envs
            episode_rewards.append(avg_reward)
            
            # Update learning rate based on performance
            self.scheduler.step(avg_reward)
            
            # Print progress with GPU stats
            if torch.cuda.is_available():
                print(f"Episode {episode + 1}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.2f}, "
                      f"CUDA Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB, "
                      f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoints
            if (episode + 1) % 100 == 0:
                self.save(f'checkpoint_episode_{episode+1}.pth')
                if torch.cuda.is_available():
                    print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

    def _train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample and prepare batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Process all data in batch
        states = torch.FloatTensor(np.vstack(states)).to(device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Compute Q values in parallel
        current_q_values = self.policy_net(states)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

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