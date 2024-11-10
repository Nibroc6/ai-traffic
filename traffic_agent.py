import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import os
from datetime import datetime
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', 
                       ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    def __init__(self, obs_size, num_lights, num_speeds):
        super(DQN, self).__init__()
        self.num_lights = num_lights
        self.num_speeds = num_speeds
        
        # Separate encoders for nodes and edges
        self.node_encoder = nn.Sequential(
            nn.Linear(2, 64),  # [num_cars, light_state]
            nn.ReLU(),
            nn.LayerNorm(64)
        ).to(device)
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(5, 64),  # [num_cars, avg_speed, speed_limit, pos_P, pos_N]
            nn.ReLU(),
            nn.LayerNorm(64)
        ).to(device)
        
        # Attention mechanism for nodes and edges
        self.node_attention = nn.MultiheadAttention(64, 4, batch_first=True).to(device)
        self.edge_attention = nn.MultiheadAttention(64, 4, batch_first=True).to(device)
        
        # Combine and process node and edge features
        self.combined_processor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        ).to(device)
        
        # Separate heads with different architectures
        self.lights_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_lights),
            nn.Sigmoid()
        ).to(device)
        
        self.speeds_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_speeds),
            nn.Softplus()  # Ensures positive speed values
        ).to(device)

    def forward(self, state_dict):
        # Process nodes
        node_features = self.node_encoder(state_dict['nodes'])
        node_features, _ = self.node_attention(node_features, node_features, node_features)
        
        # Process edges
        edge_features = self.edge_encoder(state_dict['edges'])
        edge_features, _ = self.edge_attention(edge_features, edge_features, edge_features)
        
        # Combine features using global average pooling
        node_features = torch.mean(node_features, dim=1)
        edge_features = torch.mean(edge_features, dim=1)
        
        # Concatenate and process
        combined = torch.cat([node_features, edge_features], dim=-1)
        features = self.combined_processor(combined)
        
        # Generate outputs
        lights = self.lights_head(features)
        speeds = self.speeds_head(features)
        
        return lights, speeds

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.frame = 0
    
    def push(self, *args):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.position] = Transition(*args)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.frame += 1
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
        
        # Calculate current beta
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        
        # Calculate sampling probabilities
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices and calculate importance weights
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, torch.FloatTensor(weights).to(device)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class TrafficControlAgent:
    def __init__(self, env, memory_size=100000, batch_size=256, checkpoint_dir='checkpoints'):
        self.env = env
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Calculate input sizes
        self.obs_size = (
            env.observation_space['nodes'].shape[0] * env.observation_space['nodes'].shape[1] +
            env.observation_space['edges'].shape[0] * env.observation_space['edges'].shape[1]
        )
        
        self.num_lights = env.action_space['lights'].n
        self.num_speeds = env.action_space['speeds'].shape[0]
        
        # Initialize networks
        self.policy_net = DQN(self.obs_size, self.num_lights, self.num_speeds)
        self.target_net = DQN(self.obs_size, self.num_lights, self.num_speeds)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Use Adam optimizer instead of AdaBelief
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=1e-3,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Prioritized experience replay
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Training parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update = 5
        self.gradient_clip = 1.0
        self.warm_up_steps = 1000
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'avg_losses': [],
            'best_reward': float('-inf'),
            'last_episode': 0,
            'total_steps': 0
        }

    def preprocess_state(self, state):
        """Convert state dict to tensors"""
        return {
            'nodes': torch.FloatTensor(state['nodes']).to(device),
            'edges': torch.FloatTensor(state['edges']).to(device)
        }
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return {
                'lights': self.env.action_space['lights'].sample(),
                'speeds': self.env.action_space['speeds'].sample()
            }
        
        with torch.no_grad():
            state_tensors = {
                k: torch.FloatTensor(v).unsqueeze(0).to(device)
                for k, v in state.items()
            }
            lights_pred, speeds_pred = self.policy_net(state_tensors)
            
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

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = {
            'nodes': torch.cat([s['nodes'].unsqueeze(0) for s in batch.state]),
            'edges': torch.cat([s['edges'].unsqueeze(0) for s in batch.state])
        }
        
        next_state_batch = {
            'nodes': torch.cat([s['nodes'].unsqueeze(0) for s in batch.next_state]),
            'edges': torch.cat([s['edges'].unsqueeze(0) for s in batch.next_state])
        }
        
        rewards = torch.FloatTensor(batch.reward).to(device)
        dones = torch.FloatTensor(batch.done).to(device)
        
        current_lights, current_speeds = self.policy_net(state_batch)
        
        with torch.no_grad():
            next_lights, next_speeds = self.target_net(next_state_batch)
        
        lights_target = torch.FloatTensor(np.vstack([a['lights'] for a in batch.action])).to(device)
        speeds_target = torch.FloatTensor(np.vstack([a['speeds'] for a in batch.action])).to(device)
        
        lights_td_error = nn.BCELoss(reduction='none')(current_lights, lights_target).mean(1)
        speeds_td_error = nn.MSELoss(reduction='none')(current_speeds, speeds_target).mean(1)
        
        td_error = lights_td_error + speeds_td_error
        
        self.memory.update_priorities(indices, td_error.detach().cpu().numpy())
        
        lights_loss = (weights * lights_td_error).mean()
        speeds_loss = (weights * speeds_td_error).mean()
        
        loss = lights_loss + speeds_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        
        return loss.item()

    def save_checkpoint(self, episode, rewards_history, losses, is_best=False):
        """Save a training checkpoint"""
        checkpoint = {
            'episode': episode,
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            'rewards_history': rewards_history,
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_episode_{episode}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
        
        metrics_path = os.path.join(
            self.checkpoint_dir, 
            f"metrics_episode_{episode}.json"
        )
        metrics = {
            'episode': episode,
            'epsilon': float(self.epsilon),
            'avg_reward': float(np.mean(rewards_history[-10:])),
            'avg_loss': float(np.mean(losses)) if losses else 0,
            'best_reward': float(self.training_history['best_reward'])
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Keep only last 5 checkpoints
        checkpoints = sorted([
            f for f in os.listdir(self.checkpoint_dir) 
            if f.startswith('checkpoint_episode_')
        ])
        if len(checkpoints) > 5:
            os.remove(os.path.join(self.checkpoint_dir, checkpoints[0]))
        
        print(f"Saved checkpoint at episode {episode}")

    def load_checkpoint(self, path):
        """Load a training checkpoint"""
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=device)
            
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            self.epsilon = checkpoint['epsilon']
            self.training_history = checkpoint['training_history']
            
            print(f"Loaded checkpoint from episode {checkpoint['episode']}")
            return checkpoint['episode']
        
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def train(self, num_episodes, eval_interval=1, checkpoint_interval=50, resume_path=None):
        """Train the agent with checkpointing"""
        start_episode = 0
        rewards_history = []
        
        if resume_path:
            loaded_episode = self.load_checkpoint(resume_path)
            if loaded_episode:
                start_episode = loaded_episode + 1
                print(f"Resuming training from episode {start_episode}")
        
        print(f"\nStarting training for {num_episodes} episodes:")
        print("- Eval interval:", eval_interval)
        print("- Checkpoint interval:", checkpoint_interval)
        print("- Batch size:", self.batch_size)
        print("- Memory size:", len(self.memory.buffer))
        print("- Device:", device)
        print()
        
        try:  # Add try-except to catch any errors
            for episode in range(start_episode, start_episode + num_episodes):
                print(f"\nStarting episode {episode + 1}")  # Debug log
                episode_start_time = datetime.now()
                
                print("Resetting environment...")  # Debug log
                state, _ = self.env.reset()
                total_reward = 0
                losses = []
                steps = 0
                
                # Warm-up phase (only if starting fresh)
                if episode == 0 and not resume_path:
                    print("\nStarting warm-up phase...")
                    print(f"Collecting {self.warm_up_steps} experiences with random actions...")
                    
                    for step in range(self.warm_up_steps):
                        action = self.env.action_space.sample()
                        next_state, reward, done, _, _ = self.env.step(action)
                        self.memory.push(
                            self.preprocess_state(state),
                            action,
                            reward,
                            self.preprocess_state(next_state),
                            done
                        )
                        state = next_state
                        if done:
                            state, _ = self.env.reset()
                        
                        if (step + 1) % 100 == 0:
                            print(f"Warmup progress: {step + 1}/{self.warm_up_steps} steps")
                    
                    print("Warm-up complete! Starting training...\n")
                
                # Training episode
                print("Starting episode loop...")  # Debug log
                episode_steps = 0
                
                while True:
                    if episode_steps % 100 == 0:  # Log every 100 steps
                        print(f"Episode {episode + 1} Step {episode_steps}, Total Reward: {total_reward:.2f}")
                    
                    # Select and perform action
                    print(f"  Selecting action...") if episode_steps == 0 else None
                    action = self.select_action(state)
                    
                    print(f"  Performing step...") if episode_steps == 0 else None
                    next_state, reward, done, _, info = self.env.step(action)
                    steps += 1
                    episode_steps += 1
                    
                    # Store transition
                    print(f"  Storing transition...") if episode_steps == 1 else None
                    self.memory.push(
                        self.preprocess_state(state),
                        action,
                        reward,
                        self.preprocess_state(next_state),
                        done
                    )
                    
                    total_reward += reward
                    state = next_state
                    
                    # Perform training step if enough samples
                    if len(self.memory) > self.batch_size:
                        print(f"  Training step...") if episode_steps == 1 else None
                        loss = self.train_step()
                        losses.append(loss)
                    
                    if done:
                        print(f"Episode {episode + 1} completed after {episode_steps} steps")
                        break
                    
                    # Add safety check for max steps
                    if episode_steps > 10000:  # Adjust this number based on your environment
                        print(f"WARNING: Episode {episode + 1} exceeded 10000 steps - forcing completion")
                        break
                
                # Update target network
                if episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    print(f"Updated target network")
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                # Store reward
                rewards_history.append(total_reward)
                
                # Update training history
                self.training_history['episode_rewards'].append(total_reward)
                self.training_history['avg_losses'].append(np.mean(losses) if losses else 0)
                self.training_history['last_episode'] = episode
                
                # Calculate episode time
                episode_time = datetime.now() - episode_start_time
                
                # Periodic evaluation and checkpointing
                if episode % eval_interval == 0:
                    avg_reward = np.mean(rewards_history[-eval_interval:])
                    self.scheduler.step(avg_reward)
                    
                    is_best = avg_reward > self.training_history['best_reward']
                    if is_best:
                        self.training_history['best_reward'] = avg_reward
                    
                    print(f"\nEpisode {episode + 1}/{start_episode + num_episodes} "
                          f"({episode_time.total_seconds():.1f}s)")
                    print(f"  Steps: {steps}")
                    print(f"  Reward: {total_reward:.2f}")
                    print(f"  Avg Reward (last {eval_interval}): {avg_reward:.2f}")
                    print(f"  Best Reward: {self.training_history['best_reward']:.2f}")
                    print(f"  Epsilon: {self.epsilon:.3f}")
                    print(f"  Avg Loss: {np.mean(losses):.3e}")
                    if 'crashes' in info:
                        print(f"  Crashes: {info['crashes']}")
                    if 'completed_trips' in info:
                        print(f"  Completed Trips: {info['completed_trips']}")
                    print()
                
                # Save checkpoint
                if episode % checkpoint_interval == 0:
                    self.save_checkpoint(
                        episode,
                        rewards_history,
                        losses,
                        is_best=(is_best if episode % eval_interval == 0 else False)
                    )
        
        except Exception as e:
            print(f"\nERROR during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return rewards_history
        
        # Save final checkpoint
        self.save_checkpoint(
            episode,
            rewards_history,
            losses,
            is_best=(avg_reward > self.training_history['best_reward'])
        )
        
        print("\nTraining completed!")
        print(f"Best reward achieved: {self.training_history['best_reward']:.2f}")
        
        return rewards_history

    def save(self, path):
        """Save the model to a file"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        """Load the model from a file"""
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon']