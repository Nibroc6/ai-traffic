import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

class TrafficControlEnv(gym.Env):
    def analyze_dimensions(self):
        """Add this method to TrafficControlAgent to debug dimensions"""
        print(f"Number of nodes (lights): {self.num_lights}")
        print(f"Number of edges (speeds): {self.num_speeds}")
        print(f"Total action size: {self.action_size}")
        
        # Sample observation
        state, _ = self.env.reset()
        state_tensor = torch.FloatTensor(self.preprocess_observation(state)).unsqueeze(0).to(device)
        
        # Get network output shape
        with torch.no_grad():
            action_values = self.policy_net(state_tensor)
            print(f"\nNetwork input shape: {state_tensor.shape}")
            print(f"Network output shape: {action_values.shape}")
            
        # Sample from memory
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(np.vstack(states)).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            
            print(f"\nBatch states shape: {states.shape}")
            print(f"Batch rewards shape: {rewards.shape}")
            
            # Get Q-values
            current_q_values = self.policy_net(states)
            print(f"Q-values shape: {current_q_values.shape}")

    def __init__(self, simulation, render_mode: Optional[str] = None, max_steps: int = 1000):
        super().__init__()
        self.simulation = simulation
        self.render_mode = render_mode
        self.max_steps = max_steps  # Added max_steps parameter
        self.current_step = 0       # Added step counter
        
        # Track completed trips
        self.completed_trips = 0
        self._previous_crashes = self.simulation.crashes
        
        # Define action space
        self.num_nodes = len(simulation.nodes)
        self.num_edges = len(simulation.edges)
        
        self.action_space = spaces.Dict({
            'lights': spaces.MultiBinary(self.num_nodes),
            'speeds': spaces.Box(
                low=0.1/60, 
                high=0.5/60, 
                shape=(self.num_edges,), 
                dtype=np.float32
            )
        })

        # Enhanced observation space to include car positions
        self.observation_space = spaces.Dict({
            'nodes': spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.num_nodes, 2),  # [num_cars, light_state]
                dtype=np.float32
            ),
            'edges': spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.num_edges, 5),  # [num_cars, avg_speed, speed_limit, cars_positions_P, cars_positions_N]
                dtype=np.float32
            )
        })

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset simulation state
        self.simulation.crashes = 0
        self.simulation.tot_cars = [0]
        self.completed_trips = 0
        self.current_step = 0
        self._previous_crashes = 0
        
        for node in self.simulation.nodes:
            node.cars_in_intersection = []
            node.lightud = bool(np.random.randint(0, 2))
        
        for edge in self.simulation.edges:
            edge.carsP = []
            edge.carsN = []
            edge.speed_limit = 0.2/60

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation of the environment with detailed car positions"""
        # Node observations
        node_obs = np.zeros((self.num_nodes, 2), dtype=np.float32)
        for i, node in enumerate(self.simulation.nodes):
            node_obs[i] = [
                len(node.cars_in_intersection),
                float(node.lightud)
            ]

        # Edge observations with car positions
        edge_obs = np.zeros((self.num_edges, 5), dtype=np.float32)
        for i, edge in enumerate(self.simulation.edges):
            # Calculate average positions for cars in both directions
            avg_pos_P = np.mean([car.position for car in edge.carsP]) if edge.carsP else 0
            avg_pos_N = np.mean([car.position for car in edge.carsN]) if edge.carsN else 0
            
            # Calculate average speed
            total_cars = len(edge.carsP) + len(edge.carsN)
            avg_speed = 0
            if total_cars > 0:
                speeds = ([car.speed for car in edge.carsP] + 
                         [car.speed for car in edge.carsN])
                avg_speed = sum(speeds) / total_cars
            
            edge_obs[i] = [
                total_cars,
                avg_speed,
                edge.speed_limit,
                avg_pos_P,  # Average position of cars going in positive direction
                avg_pos_N   # Average position of cars going in negative direction
            ]

        return {
            'nodes': node_obs,
            'edges': edge_obs
        }

    def _calculate_reward(self) -> float:
        """Enhanced reward calculation including completed trips"""
        # Track crashes
        new_crashes = self.simulation.crashes - self._previous_crashes
        self._previous_crashes = self.simulation.crashes
        
        # Calculate traffic flow metrics
        total_moving_cars = 0
        total_speed = 0
        edge_congestion = 0
        
        for edge in self.simulation.edges:
            cars = edge.carsP + edge.carsN
            moving_cars = [car for car in cars if car.speed > 0.01]
            total_moving_cars += len(moving_cars)
            total_speed += sum(car.speed for car in moving_cars)
            
            # Calculate congestion on edge
            capacity = edge.length * 0.1  # Assuming max 1 car per 0.1 length units
            congestion = len(cars) / capacity
            edge_congestion += max(0, congestion - 0.8)  # Only penalize if over 80% capacity
        
        # Calculate intersection congestion
        intersection_congestion = sum(
            max(0, len(node.cars_in_intersection) - self.simulation.MAX_CARS_IN_INTERSECTION + 1)
            for node in self.simulation.nodes
        )
        
        # Get number of new completed trips (by monitoring the remove_item function calls with report=True)
        new_completions = 0
        original_remove_item = self.simulation.remove_item
        def counting_remove_item(l, i, more_stuff=None, report=False):
            nonlocal new_completions
            if report:
                new_completions += 1
            return original_remove_item(l, i, more_stuff, report)
        self.simulation.remove_item = counting_remove_item
        
        # Calculate rewards
        completion_reward = 50 * new_completions    # Large reward for completed trips
        crash_penalty = -100 * new_crashes          # Heavy penalty for crashes
        
        # Flow reward based on average speed and number of moving cars
        avg_speed = total_speed / max(1, total_moving_cars)
        flow_reward = 10 * avg_speed * (total_moving_cars / max(1, total_moving_cars))
        
        # Congestion penalties
        edge_congestion_penalty = -5 * edge_congestion
        intersection_penalty = -10 * intersection_congestion
        
        # Restore original remove_item function
        self.simulation.remove_item = original_remove_item
        
        # Update total completed trips
        self.completed_trips += new_completions
        
        return (completion_reward + 
                crash_penalty + 
                flow_reward + 
                edge_congestion_penalty + 
                intersection_penalty)

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        # Apply actions
        for i, node in enumerate(self.simulation.nodes):
            node.lightud = bool(action['lights'][i].item() if isinstance(action['lights'][i], np.ndarray) else action['lights'][i])
        
        for i, edge in enumerate(self.simulation.edges):
            edge.speed_limit = float(action['speeds'][i])

        # Run simulation step
        self.simulation.tick_all()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update state
        self.current_step += 1
        observation = self._get_obs()
        info = self._get_info()
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False

        return observation, reward, terminated, truncated, info

    def _get_info(self):
        """Enhanced info dictionary"""
        return {
            'crashes': self.simulation.crashes,
            'completed_trips': self.completed_trips,
            'total_cars': self.simulation.tot_cars[0],
            'step': self.current_step
        }

    def render(self):
        """Render the environment - implementation would depend on your visualization needs"""
        pass

    def close(self):
        """Clean up environment resources"""
        pass