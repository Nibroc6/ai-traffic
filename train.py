import main
from traffic_env import TrafficControlEnv
from traffic_agent import TrafficControlAgent
import torch
import torch.multiprocessing as mp

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
    agent = TrafficControlAgent(env, batch_size=1024, memory_size=100000)
    
    # Train the agent
    print("\nStarting training...")
    try:
        agent.train(num_episodes=1000)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save("interrupted_model.pth")
        return
    
    agent.save("trained_traffic_controller.pth")
    print("Training completed")

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Required for CUDA multiprocessing
    main_training()