import main
from traffic_env import TrafficControlEnv
from traffic_agent import TrafficControlAgent
import torch
import torch.multiprocessing as mp
"""
def main_training():
    # ... previous setup code ...
    
    # Create the agent with checkpoint directory
    agent = TrafficControlAgent(env, checkpoint_dir='checkpoints')
    
    # To start fresh training with checkpoints every 50 episodes:
    agent.train(num_episodes=1000, checkpoint_interval=50)
    
    # Or to resume training from a checkpoint:
    agent.train(
        num_episodes=1000,
        checkpoint_interval=50,
        resume_path='checkpoints/checkpoint_episode_200.pth'
    )
"""
def main_training():
    # Set up CUDA for maximum performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('high')
        
        # Print CUDA information
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Create the environment
    env = TrafficControlEnv(main)
    
    # Create the agent with larger batch size
    agent = TrafficControlAgent(env, batch_size=1024, memory_size=100000, checkpoint_dir='checkpoints')
    
    # Train the agent
    print("\nStarting training...")
    try:
        agent.train(num_episodes=1000, checkpoint_interval=50)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save("interrupted_model.pth")
        return
    
    agent.save("trained_traffic_controller.pth")
    print("Training completed")

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Required for CUDA multiprocessing
    main_training()