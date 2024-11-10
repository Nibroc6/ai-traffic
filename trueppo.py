# run_model.py
import numpy as np
import os
from tensorflow import keras
from ppo_ai_model import TrafficEnv  # Make sure this points to your environment (same directory or module)

# Load the trained model from the saved file
model_filename = os.path.join(os.getcwd(), 'generic_model.keras')

if os.path.exists(model_filename):
    print(f"Loading model from {model_filename}")
    model = keras.models.load_model(model_filename)
else:
    print("Model file does not exist. Exiting.")
    exit(1)

# Initialize the environment
env = TrafficEnv()

# Calculate the size of the state from the observation space
state_size = sum(np.prod(space.shape) for space in env.observation_space.values())

# Reset the environment
state, _ = env.reset()
state = np.concatenate([np.ravel(v) for v in state.values()])
state = np.reshape(state, [1, state_size])

done = False
total_reward = 0
steps = 0

# Run the trained model in the environment for one episode
print("Running the trained model...")

while not done:
    # Predict the next action using the trained model
    action_values = model.predict(state, verbose=0)
    
    # Choose the action based on the model's output (binary vector of light states)
    action = np.round(action_values[0])  # Round the output to binary values (0 or 1 for each node)
    
    # Take a step in the environment
    next_state, reward, done, _, info = env.step(action)
    
    total_reward += reward
    steps += 1
    
    # Prepare the next state
    next_state = np.concatenate([np.ravel(v) for v in next_state.values()])
    next_state = np.reshape(next_state, [1, state_size])
    
    state = next_state
    
    # Print out progress and metrics
    print(f"Step {steps}: Reward={reward:.2f}, Goals={info['goals']}, Crashes={info['crashes']}")

# After the episode is done, print final results
print("\n=== Final Results ===")
print(f"Total reward: {total_reward:.2f}")
print(f"Goals reached: {info['goals']}")
print(f"Crashes: {info['crashes']}")
