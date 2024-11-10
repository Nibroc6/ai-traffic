from gymnasium import Env
from gymnasium.spaces import MultiBinary, Dict, Box, Discrete
import numpy as np
import random
from tensorflow import keras
from collections import deque
import tensorflow as tf
import os

from datetime import datetime

import time

from main import *

# Limit TensorFlow to better handle CPU operations
tf.config.threading.set_inter_op_parallelism_threads(16)
tf.config.threading.set_intra_op_parallelism_threads(16)


class TrafficEnv(Env):
    def __init__(self):
        super().__init__()
        self.num_nodes = len(nodes)
        self.action_space = MultiBinary(self.num_nodes)
        
        self.observation_space = Dict({
            "Cars_Entering": Box(low=0, high=1, shape=(self.num_nodes, 2), dtype=np.float32),
            "Cars_Exiting": Box(low=0, high=1, shape=(self.num_nodes, 2), dtype=np.float32),
            "Speed_Entering": Box(low=0, high=1, shape=(self.num_nodes, 2), dtype=np.float32),
            "Average_Speed_Exiting": Box(low=0, high=1, shape=(self.num_nodes, 2), dtype=np.float32),
            "Light_States": Box(low=0, high=1, shape=(self.num_nodes, 2), dtype=np.float32)
        })
        
        self.crash_count = 0
        self.goals_reached = 0
        self.reset()
    def set_crashes(self, value):
        # Debugging: print the new crash value
        self.crash_count = value
    
    def set_goals_reached(self, value):
        # Debugging: print the new goals reached value
        self.goals_reached = value
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        for e in edges:
            e.carsP.clear()
            e.carsN.clear()
            
            length = e.length
            if random.randint(0, 1):
                c = car(e.nodeP, e.nodeN)
                c.position = random.uniform(0, length)
                e.carsP.insert(0, c)
                
                c = car(e.nodeN, e.nodeP)
                c.position = random.uniform(0, length)
                e.carsN.insert(0, c)
        
        for node in nodes:
            node.lightud = False
        
        # Resetting crash and goal counts using setters
        self.set_crashes(0)
        self.set_goals_reached(0)
        set_crashes(0)
        set_goals_reached(0)
        return self._get_observation(), {}
    
    def step(self, action):
        new_crash_count = get_crashes()
        new_goals_reached = get_successes()
        
        #node_idx, light_state = self._decode_action(action)
        #nodes[node_idx].lightud = light_state
        
        tick_all()

        for i in range(len(nodes)):
            if(nodes[i].secondPassed()):
                nodes[i].lightud = bool(action[i])
                nodes[i].lastTime = time.time()
                
        new_crashes = new_crash_count - self.crash_count
        new_goals = new_goals_reached - self.goals_reached
        
        # Set crashes and goals reached using setters
        self.set_crashes(self.crash_count)
        self.set_goals_reached(self.goals_reached)
        self.crash_count = new_crash_count
        self.goals_reached = new_goals_reached
        
        reward = new_goals * 10 - new_crashes * 20
        
        done = self.goals_reached >= 100 or self.crash_count >= 50
        
        return self._get_observation(), reward, done, False, {
            'goals': self.goals_reached,
            'crashes': self.crash_count
        }
        '''
            def _decode_action(self, action):
                print("to decode", action)
                node_idx = action // 2
                light_state = bool(action % 2)
                return node_idx, light_state
        '''

    def _get_observation(self):
        cars_entering = np.zeros((self.num_nodes, 2), dtype=np.float32)
        cars_exiting = np.zeros((self.num_nodes, 2), dtype=np.float32)
        speed_entering = np.zeros((self.num_nodes, 2), dtype=np.float32)
        speed_exiting = np.zeros((self.num_nodes, 2), dtype=np.float32)
        light_states = np.zeros((self.num_nodes, 2), dtype=np.float32)
        
        # Generate observations for each node
        for i, node in enumerate(nodes):
            light_states[i] = [float(node.lightud), float(not node.lightud)]
            
            # Update entering and exiting car counts
            edges_ud = [e for e in edges if (e.nodeP == node or e.nodeN == node) and (node.edges['d'] in [e.nodeN, e.nodeP] or node.edges['u'] in [e.nodeN, e.nodeP])]
            edges_lr = [e for e in edges if (e.nodeP == node or e.nodeN == node) and (node.edges['l'] in [e.nodeN, e.nodeP] or node.edges['r'] in [e.nodeN, e.nodeP])]
            
            if edges_ud:
                total_cars_ud = sum(len(e.carsP) + len(e.carsN) for e in edges_ud)
                cars_entering[i][0] = min(total_cars_ud / (len(edges_ud) * 10), 1.0)
            
            if edges_lr:
                total_cars_lr = sum(len(e.carsP) + len(e.carsN) for e in edges_lr)
                cars_entering[i][1] = min(total_cars_lr / (len(edges_lr) * 10), 1.0)

        # Add a debug print to check if the observation is correctly generated

        
        return {
            "Cars_Entering": cars_entering,
            "Cars_Exiting": cars_exiting,
            "Speed_Entering": speed_entering,
            "Average_Speed_Exiting": speed_exiting,
            "Light_States": light_states
        }

    






class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Reduced memory size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            # If exploring, return a random binary vector (size equal to number of nodes)
            return np.random.randint(2, size=self.action_size)  # Random 0s and 1s vector of length `self.action_size`
        
        # Predict the action values using the model
        act_values = self.model.predict(state, verbose=0)

        # Convert the predicted values to a binary vector (0 or 1) for each node
        return np.round(act_values[0])  # Round to 0 or 1 for each node


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])  # Actions are binary vectors
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            # Convert the binary action vector to an index (the node where the light is turned on)
            action_idx = np.argmax(actions[i])  # Get the index where action is 1 (light turned on)
            
            if dones[i]:
                # If done, directly assign reward to that action's Q-value
                targets[i][action_idx] = rewards[i]
            else:
                # Otherwise, apply the Bellman equation
                targets[i][action_idx] = rewards[i] + self.gamma * np.amax(target_next[i])

        # Fit the model using the updated Q-values
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

        # Decay epsilon after each episode
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

env = TrafficEnv()
state_size = sum(np.prod(space.shape) for space in env.observation_space.values())
action_size = env.action_space.n

print("\n=== Training Configuration ===")
print(f"State size: {state_size}")
print(f"Action size: {action_size}")
print(f"Training start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

agent = DQNAgent(state_size=state_size, action_size=action_size)
batch_size = 64
episodes = 100

# Training metrics
total_rewards = []
moving_avg_rewards = []
best_episode_reward = float('-inf')
start_time = time.time()

# Training loop
for episode in range(episodes):
    episode_start = time.time()
    state, _ = env.reset()
    state = np.concatenate([np.ravel(v) for v in state.values()])
    state = np.reshape(state, [1, state_size])
    
    episode_reward = 0
    steps = 0
    
    print(f"\nEpisode {episode + 1}/{episodes} Starting...")
    
    for time_step in range(1000):
        steps += 1
        action = agent.act(state)
        next_state, reward, done, _, info = env.step(action)
        episode_reward += reward
        
        next_state = np.concatenate([np.ravel(v) for v in next_state.values()])
        next_state = np.reshape(next_state, [1, state_size])
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
        # Print progress every 100 steps
        if time_step % 100 == 0:
            print(f"  Step {time_step}: Goals={info['goals']}, Crashes={info['crashes']}")
        
        if done:
            break
    
    # Episode summary
    episode_duration = time.time() - episode_start
    total_rewards.append(episode_reward)
    moving_avg = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
    moving_avg_rewards.append(moving_avg)
    
    if episode_reward > best_episode_reward:
        best_episode_reward = episode_reward
    
    print("\n=== Episode Summary ===")
    print(f"Episode: {episode + 1}/{episodes}")
    print(f"Duration: {episode_duration:.2f} seconds")
    print(f"Steps taken: {steps}")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Moving avg (100 ep): {moving_avg:.2f}")
    print(f"Goals reached: {info['goals']}")
    print(f"Crashes: {info['crashes']}")
    print(f"Epsilon: {agent.epsilon:.3f}")
    print(f"Memory size: {len(agent.memory)}")
    print("=" * 50)
    model_filename = os.path.join(os.getcwd(), 'generic_model.keras')
    agent.model.save(model_filename)
    # Update target network periodically
    if episode % 10 == 0:
        agent.update_target_model()
        print("\nTarget model updated")
    
    # Print overall statistics every 100 episodes
    if (episode + 1) % 100 == 0:
        elapsed_time = time.time() - start_time
        print("\n=== Training Progress ===")
        print(f"Time elapsed: {elapsed_time/3600:.2f} hours")
        print(f"Best episode reward: {best_episode_reward:.2f}")
        print(f"Current moving average: {moving_avg:.2f}")
        print(f"Memory usage: {len(agent.memory)}/{agent.memory.maxlen}")
        print("=" * 50)
        



# Final training summary
print("\n=== Training Complete ===")
print(f"Total training time: {(time.time() - start_time)/3600:.2f} hours")
print(f"Best episode reward: {best_episode_reward:.2f}")
print(f"Final moving average: {moving_avg_rewards[-1]:.2f}")
print(f"Final epsilon: {agent.epsilon:.3f}")
print("=" * 50)