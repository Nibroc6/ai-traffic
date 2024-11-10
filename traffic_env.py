import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

class TrafficControlEnv(gym.Env):
    """Custom Environment for traffic control that follows gym interface"""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, simulation, render_mode: Optional[str] = None):
        super().__init__()
        self.simulation = simulation  # Reference to the traffic simulation
        self.render_mode = render_mode
        
        # Define action space
        # For each node: light state (binary)
        # For each edge: speed limit (continuous between 0.1 and 0.5)
        self.num_nodes = len(simulation.nodes)
        self.num_edges = len(simulation.edges)
        
        # Combined action space for lights and speed limits
        self.action_space = spaces.Dict({
            'lights': spaces.MultiBinary(self.num_nodes),
            'speeds': spaces.Box(
                low=0.1/60, 
                high=0.5/60, 
                shape=(self.num_edges,), 
                dtype=np.float32
            )
        })

        # Define observation space
        # For each node: number of cars in intersection, light state
        # For each edge: number of cars, average speed, current speed limit
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
                shape=(self.num_edges, 3),  # [num_cars, avg_speed, speed_limit]
                dtype=np.float32
            )
        })

        self.current_step = 0
        self.max_steps = 1000
        self._previous_crashes = self.simulation.crashes

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation of the environment"""
        # Node observations
        node_obs = np.zeros((self.num_nodes, 2), dtype=np.float32)
        for i, node in enumerate(self.simulation.nodes):
            node_obs[i] = [
                len(node.cars_in_intersection),
                float(node.lightud)
            ]

        # Edge observations
        edge_obs = np.zeros((self.num_edges, 3), dtype=np.float32)
        for i, edge in enumerate(self.simulation.edges):
            total_cars = len(edge.carsP) + len(edge.carsN)
            avg_speed = 0
            if total_cars > 0:
                speeds = ([car.speed for car in edge.carsP] + 
                         [car.speed for car in edge.carsN])
                avg_speed = sum(speeds) / total_cars
            
            edge_obs[i] = [
                total_cars,
                avg_speed,
                edge.speed_limit
            ]

        return {
            'nodes': node_obs,
            'edges': edge_obs
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment"""
        return {
            'crashes': self.simulation.crashes,
            'total_cars': self.simulation.tot_cars[0],
            'step': self.current_step
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset simulation state
        self.simulation.crashes = 0
        self.simulation.tot_cars = [0]
        for node in self.simulation.nodes:
            node.cars_in_intersection = []
            node.lightud = bool(np.random.randint(0, 2))
        
        for edge in self.simulation.edges:
            edge.carsP = []
            edge.carsN = []
            edge.speed_limit = 0.2/60

        self.current_step = 0
        self._previous_crashes = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        # Apply actions
        # Update light states
        for i, node in enumerate(self.simulation.nodes):
            # Convert to scalar boolean using item()
            node.lightud = bool(action['lights'][i].item() if isinstance(action['lights'][i], np.ndarray) else action['lights'][i])
        
        # Update speed limits
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

    def _calculate_reward(self) -> float:
        """Calculate the reward for the current step"""
        # Components for reward calculation
        new_crashes = self.simulation.crashes - self._previous_crashes
        self._previous_crashes = self.simulation.crashes
        
        # Calculate average traffic flow
        total_moving_cars = 0
        total_speed = 0
        for edge in self.simulation.edges:
            cars = edge.carsP + edge.carsN
            total_moving_cars += len(cars)
            total_speed += sum(car.speed for car in cars)
        
        avg_speed = total_speed / max(1, total_moving_cars)
        
        # Penalties and rewards
        crash_penalty = -100 * new_crashes  # Heavy penalty for crashes
        flow_reward = 10 * avg_speed  # Reward for maintaining traffic flow
        congestion_penalty = -0.1 * sum(
            len(node.cars_in_intersection) for node in self.simulation.nodes
        )  # Small penalty for congestion
        
        return crash_penalty + flow_reward + congestion_penalty

    def render(self):
        """Render the environment - implementation would depend on your visualization needs"""
        pass

    def close(self):
        """Clean up environment resources"""
        pass