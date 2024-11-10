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
        self.remove_item = remove_item  # Add the remove_item function
        
    def tick_all(self):
        tick_all()  # Call the imported tick_all function
        
    def get_crashes(self):
        return get_crashes()
        
    def get_successes(self):
        return get_successes()
        
    def set_crashes(self, value):
        set_crashes(value)
        self.crashes = value  # Update local value too
        
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
env = TrafficControlEnv(simulation)
agent = TrafficControlAgent(env)

# Load the trained model
MODEL_PATH = "./checkpoints/best_model.pth"  # Update this path to your model
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
    bg.fill((200, 255, 200))  # Light green color

# Node radius calculation
if screen.get_width() >= screen.get_height():
    node_rad = (screen.get_width() / mapsize[0]) / 10
else:
    node_rad = (screen.get_height() / mapsize[1]) / 10

# Road width
road_width = 15

# Car dimensions
carHeight = 10
carWidth = 5

# Initialize fonts
pygame.font.init()
my_font = pygame.font.SysFont('Arial Rounded MT Bold', 70)
node_font = pygame.font.SysFont('Comic Sans', 10)
info_font = pygame.font.SysFont('Arial', 20)

# Reset environment
state, _ = env.reset()

# Function to get model actions
def get_model_action(env):
    observation = env._get_obs()
    with torch.no_grad():
        return agent.select_action(observation)

# Initialize fps counter
fps_counter = 0
fps_timer = 0
fps = 0

# Main game loop
while running:
    # Get action from model
    action = get_model_action(env)
    
    # Apply action and step environment
    observation, reward, done, _, info = env.step(action)
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # Reset on 'R' key
                state, _ = env.reset()

    # Update crashes/successes text
    text_surface = my_font.render('Crashes: ' + str(int(simulation.get_crashes()/2)), False, (11, 15, 106))
    text_surface_s = my_font.render('Successes: ' + str(simulation.get_successes()), False, (11, 15, 106))
    
    # Add model info text
    reward_text = info_font.render(f'Reward: {reward:.2f}', False, (11, 15, 106))
    step_text = info_font.render(f'Step: {env.current_step}', False, (11, 15, 106))

    # Clear the screen
    screen.fill("white")

    # Draw background
    s = max(screen.get_width(), screen.get_height())
    for i in range(0, s // bg.get_size()[0] + 1):
        for f in range(0, s // bg.get_size()[1] + 1):
            screen.blit(bg, (i * bg.get_size()[0], f * bg.get_size()[1]))

    # Loop through nodes to render roads and cars
    for n in nodes:
        node_pos = pygame.Vector2(
            ((screen.get_width() / mapsize[0]) * n.x) + node_rad * 5,
            ((screen.get_height() / mapsize[1]) * n.y) + node_rad * 5
        )

        # Yellow line offset and width
        offset = int(road_width / 8)
        yWidth = int(road_width / 9)

        # Iterate through each direction
        for direction in ["u", "d", "l", "r"]:
            if n.edges[direction]:
                e = n.edges[direction]
                a = e.nodeP
                b = e.nodeN

                # Calculate node positions in screen coordinates
                nodeP_screen_x = ((screen.get_width() / mapsize[0]) * a.x) + node_rad * 5
                nodeP_screen_y = ((screen.get_height() / mapsize[1]) * a.y) + node_rad * 5
                nodeN_screen_x = ((screen.get_width() / mapsize[0]) * b.x) + node_rad * 5
                nodeN_screen_y = ((screen.get_height() / mapsize[1]) * b.y) + node_rad * 5

                delta_x = nodeN_screen_x - nodeP_screen_x
                delta_y = nodeN_screen_y - nodeP_screen_y

                # Render road
                if a != n:
                    prev_node_pos = pygame.Vector2(nodeP_screen_x, nodeP_screen_y)
                    pygame.draw.line(screen, "black", node_pos, prev_node_pos, width=road_width)
                else:
                    prev_node_pos = pygame.Vector2(nodeN_screen_x, nodeN_screen_y)
                    pygame.draw.line(screen, "black", node_pos, prev_node_pos, width=road_width)

                # Draw yellow lines (lane dividers)
                if direction in ["u", "d"]:
                    # Vertical roads
                    lineStart = pygame.Vector2(prev_node_pos.x + offset, prev_node_pos.y)
                    lineEnd = pygame.Vector2(node_pos.x + offset, node_pos.y)
                    pygame.draw.line(screen, "yellow", lineStart, lineEnd, width=yWidth)
                    lineStart = pygame.Vector2(prev_node_pos.x - offset, prev_node_pos.y)
                    lineEnd = pygame.Vector2(node_pos.x - offset, node_pos.y)
                    pygame.draw.line(screen, "yellow", lineStart, lineEnd, width=yWidth)
                else:
                    # Horizontal roads
                    lineStart = pygame.Vector2(prev_node_pos.x, prev_node_pos.y + offset)
                    lineEnd = pygame.Vector2(node_pos.x, node_pos.y + offset)
                    pygame.draw.line(screen, "yellow", lineStart, lineEnd, width=yWidth)
                    lineStart = pygame.Vector2(prev_node_pos.x, prev_node_pos.y - offset)
                    lineEnd = pygame.Vector2(node_pos.x, node_pos.y - offset)
                    pygame.draw.line(screen, "yellow", lineStart, lineEnd, width=yWidth)

                # Render cars on the edge
                for c in e.carsP:
                    fraction = c.position / e.length
                    car_x = nodeP_screen_x + fraction * delta_x
                    car_y = nodeP_screen_y + fraction * delta_y

                    if direction in ["u", "d"]:
                        car_rect = pygame.Rect(car_x + offset, car_y, carWidth, carHeight)
                    else:
                        car_rect = pygame.Rect(car_x, car_y + offset, carHeight, carWidth)
                    
                    pygame.draw.rect(screen, pygame.Color(c.Color), car_rect)

                for c in e.carsN:
                    fraction = c.position / e.length
                    car_x = nodeN_screen_x - fraction * delta_x
                    car_y = nodeN_screen_y - fraction * delta_y

                    if direction in ["u", "d"]:
                        car_rect = pygame.Rect(car_x - carWidth - offset, car_y, carWidth, carHeight)
                    else:
                        car_rect = pygame.Rect(car_x, car_y - carWidth - offset, carHeight, carWidth)

                    pygame.draw.rect(screen, pygame.Color(c.Color), car_rect)

    # Draw traffic lights at nodes
    for n in nodes:
        node_pos = pygame.Vector2(
            ((screen.get_width() / mapsize[0]) * n.x) + node_rad * 5,
            ((screen.get_height() / mapsize[1]) * n.y) + node_rad * 5
        )
        # node_color = "green" if n.lightud else "red"
        # pygame.draw.circle(screen, node_color, node_pos, node_rad)
        if(n.lightud):
            virtColor = "green"
            horColor = "red"
        else:
            virtColor = "red"
            horColor = "green"
        pygame.draw.polygon(screen, virtColor, ((node_pos.x-node_rad,node_pos.y),(node_pos.x,node_pos.y-2*node_rad),(node_pos.x+node_rad,node_pos.y)))
        pygame.draw.polygon(screen, virtColor, ((node_pos.x+node_rad,node_pos.y),(node_pos.x,node_pos.y+2*node_rad),(node_pos.x-node_rad,node_pos.y)))
        
        pygame.draw.polygon(screen, horColor, ((node_pos.x-2*node_rad,node_pos.y),(node_pos.x,node_pos.y-node_rad),(node_pos.x,node_pos.y+node_rad)))
        pygame.draw.polygon(screen, horColor, ((node_pos.x+2*node_rad,node_pos.y),(node_pos.x,node_pos.y+node_rad),(node_pos.x,node_pos.y-node_rad)))
        #counter
        t_surface = node_font.render(str(len(n.cars_in_intersection)), False, (11, 15, 106))
        screen.blit(t_surface, (node_pos.x-.5*node_rad,node_pos.y-1.5*node_rad))

    # Draw stats
    screen.blit(text_surface, (0,0))  # Crashes
    screen.blit(text_surface_s, (0,50))  # Successes
    screen.blit(reward_text, (0,120))  # Reward
    screen.blit(step_text, (0,150))  # Step
    
    # Calculate and display FPS
    fps_counter += 1
    fps_timer += dt
    if fps_timer >= 1.0:  # Update FPS every second
        fps = fps_counter
        fps_counter = 0
        fps_timer = 0
    fps_text = info_font.render(f'FPS: {fps}', False, (11, 15, 106))
    screen.blit(fps_text, (0,180))
    
    # Update the display
    pygame.display.flip()

    # Limit FPS
    dt = clock.tick(60) / 1000

    # Reset if episode is done
    if done:
        state, _ = env.reset()

pygame.quit()