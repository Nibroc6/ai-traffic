import numpy as np
from tensorflow.keras.models import load_model

class TrafficControlAgent:
    def __init__(self, model_path, state_size, action_size):
        # Load the trained model from the specified file path
        self.model = load_model(model_path)
        
        # Set the state and action sizes
        self.state_size = state_size
        self.action_size = action_size

    def predict_action(self, state):
        """
        Given a state, predict the next action using the trained model.
        
        Parameters:
        - state: The current state of the environment, which should be a 1D array of shape (state_size,)
        
        Returns:
        - action: The predicted action, which is a binary vector (0 or 1 for each node)
        """
        # Reshape the state into the required format for prediction
        state = np.reshape(state, [1, self.state_size])

        # Predict the action values using the model
        action_values = self.model.predict(state, verbose=0)

        # Return the action as a binary vector (0 or 1 for each node)
        return np.round(action_values[0])

# Example of how to use the TrafficControlAgent class

# Assuming `state_size` and `action_size` are the same as in the training script
state_size = 16  # Example value, replace with the actual size from the training environment
action_size = 8  # Example value, replace with the actual action size from the training environment
model_path = 'connor_model.keras'  # Path to the saved model

# Initialize the TrafficControlAgent with the trained model
agent = TrafficControlAgent(model_path=model_path, state_size=state_size, action_size=action_size)

# Example state (replace with actual state from the environment)
state = np.random.rand(state_size)  # Random example, replace with the actual state

# Predict the next action based on the current state
action = agent.predict_action(state)
print(f"Predicted Action: {action}")

