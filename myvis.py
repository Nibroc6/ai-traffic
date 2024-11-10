import pygame
import random
import torch
from main import (nodes, edges, mapsize, get_crashes, get_successes, 
                 car_breaking_range, crash_dist, traffic_light_range, 
                 max_time_in_intersection, tick_all, set_crashes,
                 set_goals_reached, remove_item)
from traffic_env import TrafficControlEnv
from traffic_agent import TrafficControlAgent

# Create a simulation namespace object that includes all required functionality
class SimulationNamespace:
    def __init__(self):
        self.nodes = nodes
        self.edges = edges
        self.crashes = 0
        self.tot_cars = [0]
        self.MAX_CARS_IN_INTERSECTION = 1
        self.remove_item = remove_item
        
    def tick_all(self):
        tick_all()
        
    def get_crashes(self):
        return get_crashes()
        
    def get_successes(self):
        return get_successes()
        
    def set_crashes(self, value):
        set_crashes(value)
        self.crashes = value
        
    def set_goals_reached(self, value):
        set_goals_reached(value)

simulation = SimulationNamespace()

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()
running = True
dt = 0

# Initialize environment and agent
env = TrafficControlEnv(simulation, max_steps=float('inf'))  # Set infinite max steps
agent = TrafficControlAgent(env)

# Load the trained model
MODEL_PATH = "./checkpoints/best_model.pth"
try:
    agent.load(MODEL_PATH)
    print("Loaded trained model successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Running with random actions")

# Background image
try:
    bgOriginal = pygame.image.load("grass.png")
    bg = pygame.transform.scale(bgOriginal, (80, 80))
except pygame.error:
    print("Warning: grass.png not found, using solid color background")
    bg = pygame.Surface((80, 80))
    bg.fill((200, 255, 200))

# Calculate dimensions
node_rad = (screen.get_width() / mapsize[0]) / 10 if screen.get_width() >= screen.get_height() else (screen.get_height() / mapsize[1]) / 10
road_width = 15
carHeight = 10
carWidth = 5

# Initialize fonts
pygame.font.init()
my_font = pygame.font.SysFont('Arial Rounded MT Bold', 70)
node_font = pygame.font.SysFont('Comic Sans', 10)
info_font = pygame.font.SysFont('Arial', 20)

# Initialize metrics
total_reward = 0
steps = 0
fps_counter = 0
fps_timer = 0
fps = 0

# Initial environment setup
state, _ = env.reset()

# Function to get model actions
def get_model_action(env):
    observation = env._get_obs()
    with torch.no_grad():
        return agent.select_action(observation)

# Main game loop
while running:
    # Get and apply model action
    action = get_model_action(env)
    observation, reward, _, _, info = env.step(action)  # Ignore done signal
    
    # Update metrics
    total_reward += reward
    steps += 1
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # Manual reset
                state, _ = env.reset()
                total_reward = 0
                steps = 0
                simulation.set_crashes(0)
                simulation.set_goals_reached(0)

    # Clear and draw background
    screen.fill("white")
    s = max(screen.get_width(), screen.get_height())
    for i in range(0, s // bg.get_size()[0] + 1):
        for f in range(0, s // bg.get_size()[1] + 1):
            screen.blit(bg, (i * bg.get_size()[0], f * bg.get_size()[1]))

    # Draw roads and intersections
    for n in nodes:
        node_pos = pygame.Vector2(
            ((screen.get_width() / mapsize[0]) * n.x) + node_rad * 5,
            ((screen.get_height() / mapsize[1]) * n.y) + node_rad * 5
        )

        # Road rendering
        offset = int(road_width / 8)
        yWidth = int(road_width / 9)

        for direction in ["u", "d", "l", "r"]:
            if n.edges[direction]:
                e = n.edges[direction]
                a = e.nodeP
                b = e.nodeN

                # Calculate positions
                nodeP_screen_x = ((screen.get_width() / mapsize[0]) * a.x) + node_rad * 5
                nodeP_screen_y = ((screen.get_height() / mapsize[1]) * a.y) + node_rad * 5
                nodeN_screen_x = ((screen.get_width() / mapsize[0]) * b.x) + node_rad * 5
                nodeN_screen_y = ((screen.get_height() / mapsize[1]) * b.y) + node_rad * 5

                delta_x = nodeN_screen_x - nodeP_screen_x
                delta_y = nodeN_screen_y - nodeP_screen_y

                # Draw road
                prev_node_pos = pygame.Vector2(nodeP_screen_x if a != n else nodeN_screen_x,
                                             nodeP_screen_y if a != n else nodeN_screen_y)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width=road_width)

                # Draw lane dividers
                if direction in ["u", "d"]:
                    for off in [offset, -offset]:
                        lineStart = pygame.Vector2(prev_node_pos.x + off, prev_node_pos.y)
                        lineEnd = pygame.Vector2(node_pos.x + off, node_pos.y)
                        pygame.draw.line(screen, "yellow", lineStart, lineEnd, width=yWidth)
                else:
                    for off in [offset, -offset]:
                        lineStart = pygame.Vector2(prev_node_pos.x, prev_node_pos.y + off)
                        lineEnd = pygame.Vector2(node_pos.x, node_pos.y + off)
                        pygame.draw.line(screen, "yellow", lineStart, lineEnd, width=yWidth)

                # Draw cars
                for c, is_positive in [(c, True) for c in e.carsP] + [(c, False) for c in e.carsN]:
                    fraction = c.position / e.length
                    if is_positive:
                        car_x = nodeP_screen_x + fraction * delta_x
                        car_y = nodeP_screen_y + fraction * delta_y
                    else:
                        car_x = nodeN_screen_x - fraction * delta_x
                        car_y = nodeN_screen_y - fraction * delta_y

                    if direction in ["u", "d"]:
                        car_rect = pygame.Rect(
                            car_x + (offset if is_positive else -carWidth - offset),
                            car_y,
                            carWidth,
                            carHeight
                        )
                    else:
                        car_rect = pygame.Rect(
                            car_x,
                            car_y + (offset if is_positive else -carWidth - offset),
                            carHeight,
                            carWidth
                        )
                    
                    pygame.draw.rect(screen, pygame.Color(c.Color), car_rect)

    # Draw traffic lights and intersection counts
    for n in nodes:
        node_pos = pygame.Vector2(
            ((screen.get_width() / mapsize[0]) * n.x) + node_rad * 5,
            ((screen.get_height() / mapsize[1]) * n.y) + node_rad * 5
        )
        pygame.draw.circle(screen, "green" if n.lightud else "red", node_pos, node_rad)
        t_surface = node_font.render(str(len(n.cars_in_intersection)), False, (11, 15, 106))
        screen.blit(t_surface, (node_pos.x-.5*node_rad, node_pos.y-1.5*node_rad))

    # Update FPS counter
    fps_counter += 1
    fps_timer += dt
    if fps_timer >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_timer = 0

    # Draw stats
    stats_texts = [
        (my_font, f'Crashes: {int(simulation.get_crashes()/2)}', (0, 0)),
        (my_font, f'Successes: {simulation.get_successes()}', (0, 50)),
        (info_font, f'Reward: {reward:.2f}', (0, 120)),
        (info_font, f'Total Reward: {total_reward:.2f}', (0, 150)),
        (info_font, f'Steps: {steps}', (0, 180)),
        (info_font, f'FPS: {fps}', (0, 210))
    ]
    
    for font, text, pos in stats_texts:
        surface = font.render(text, False, (11, 15, 106))
        screen.blit(surface, pos)

    # Update display and control frame rate
    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()