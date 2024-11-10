import pygame
import numpy as np
from tensorflow import keras
from main import *
import time

class ModelVisualizer:
    def __init__(self):
        # Load the trained model
        try:
            self.model = keras.models.load_model('generic_model (1).keras')
            print("Successfully loaded trained model")
            print(f"Expected input shape: {self.model.input_shape}")
            self.using_model = True
        except Exception as e:
            print(f"Could not load model: {e}")
            self.using_model = False
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Load and scale background
        self.bgOriginal = pygame.image.load("grass.png")
        self.bg = pygame.transform.scale(self.bgOriginal, (80, 80))
        
        # Calculate display parameters
        self.node_rad = (self.screen.get_width() / mapsize[0]) / 10 if self.screen.get_width() >= self.screen.get_height() else (self.screen.get_height() / mapsize[1]) / 10
        self.road_width = 15
        self.carHeight = 10
        self.carWidth = 5
        
        # Initialize fonts
        pygame.font.init()
        self.my_font = pygame.font.SysFont('Arial Rounded MT Bold', 70)
        self.node_font = pygame.font.SysFont('Comic Sans', 10)
        self.info_font = pygame.font.SysFont('Arial', 20)
        
        # Timer for model/random updates
        self.time_elapsed_since_last_action = 0
        self.update_interval = 3  # Seconds between light changes
        
        # Number of nodes
        self.num_nodes = len(nodes)
    
    def _get_model_observation(self):
        """Generate observation state for the model with correct dimensionality"""
        # Initialize observation arrays with correct shapes
        observation = {
            "Cars_Entering": np.zeros((self.num_nodes, 2), dtype=np.float32),
            "Cars_Exiting": np.zeros((self.num_nodes, 2), dtype=np.float32),
            "Speed_Entering": np.zeros((self.num_nodes, 2), dtype=np.float32),
            "Average_Speed_Exiting": np.zeros((self.num_nodes, 2), dtype=np.float32),
            "Light_States": np.zeros((self.num_nodes, 2), dtype=np.float32)
        }
        
        # Generate observations for each node
        for i, node in enumerate(nodes):
            # Light states (unchanged)
            observation["Light_States"][i] = [float(node.lightud), float(not node.lightud)]
            
            # Calculate cars and speeds for vertical traffic (up/down)
            edges_ud = []
            if node.edges['u'] and isinstance(node.edges['u'], edge):
                edges_ud.append(node.edges['u'])
            if node.edges['d'] and isinstance(node.edges['d'], edge):
                edges_ud.append(node.edges['d'])
                
            # Calculate cars and speeds for horizontal traffic (left/right)
            edges_lr = []
            if node.edges['l'] and isinstance(node.edges['l'], edge):
                edges_lr.append(node.edges['l'])
            if node.edges['r'] and isinstance(node.edges['r'], edge):
                edges_lr.append(node.edges['r'])
            
            # Process vertical edges
            if edges_ud:
                cars_entering_ud = 0
                cars_exiting_ud = 0
                speed_entering_ud = 0
                speed_exiting_ud = 0
                total_cars_ud = 0
                
                for e in edges_ud:
                    # Count cars approaching the intersection
                    if e.nodeN == node:
                        cars_entering_ud += len(e.carsP)
                        speed_entering_ud += sum(c.speed for c in e.carsP)
                    if e.nodeP == node:
                        cars_entering_ud += len(e.carsN)
                        speed_entering_ud += sum(c.speed for c in e.carsN)
                    total_cars_ud += len(e.carsP) + len(e.carsN)
                
                observation["Cars_Entering"][i][0] = min(cars_entering_ud / (len(edges_ud) * 10), 1.0)
                observation["Cars_Exiting"][i][0] = min(cars_exiting_ud / (len(edges_ud) * 10), 1.0)
                observation["Speed_Entering"][i][0] = min(speed_entering_ud / (total_cars_ud + 1), 1.0)
                observation["Average_Speed_Exiting"][i][0] = min(speed_exiting_ud / (total_cars_ud + 1), 1.0)
            
            # Process horizontal edges
            if edges_lr:
                cars_entering_lr = 0
                cars_exiting_lr = 0
                speed_entering_lr = 0
                speed_exiting_lr = 0
                total_cars_lr = 0
                
                for e in edges_lr:
                    # Count cars approaching the intersection
                    if e.nodeN == node:
                        cars_entering_lr += len(e.carsP)
                        speed_entering_lr += sum(c.speed for c in e.carsP)
                    if e.nodeP == node:
                        cars_entering_lr += len(e.carsN)
                        speed_entering_lr += sum(c.speed for c in e.carsN)
                    total_cars_lr += len(e.carsP) + len(e.carsN)
                
                observation["Cars_Entering"][i][1] = min(cars_entering_lr / (len(edges_lr) * 10), 1.0)
                observation["Cars_Exiting"][i][1] = min(cars_exiting_lr / (len(edges_lr) * 10), 1.0)
                observation["Speed_Entering"][i][1] = min(speed_entering_lr / (total_cars_lr + 1), 1.0)
                observation["Average_Speed_Exiting"][i][1] = min(speed_exiting_lr / (total_cars_lr + 1), 1.0)
        
        # Flatten and concatenate all observation values
        flat_observation = np.concatenate([arr.flatten() for arr in observation.values()])
        
        # Debug print to verify observation changes
        print(f"Observation shape: {flat_observation.shape}")
        print(f"Non-zero elements: {np.count_nonzero(flat_observation)}")
        
        return np.reshape(flat_observation, [1, -1])
    
    def update_traffic_lights(self):
        if self.using_model:
            try:
                observation = self._get_model_observation()
                prediction = self.model.predict(observation, verbose=0)
                print(prediction)
                action = np.argmax(prediction[0])
                
                # Debug prints
                print(f"Model prediction shape: {prediction.shape}")
                print(f"Selected action: {action}")
                
                # Decode action (same as in training environment)
                node_idx = action // 2
                light_state = bool(action % 2)
                
                print(f"Updating node {node_idx} to state {light_state}")
                
                # Update specific node's light
                if 0 <= node_idx < len(nodes):
                    nodes[node_idx].lightud = light_state
                else:
                    print(f"Invalid node index: {node_idx}")
            except Exception as e:
                print(f"Error in model prediction: {e}")
                self.using_model = False
        else:
            # Random light changes for all nodes
            for n in nodes:
                n.lightud = not n.lightud
    
    def draw_roads_and_cars(self):
        for n in nodes:
            n.ctick()
            node_pos = pygame.Vector2(
                ((self.screen.get_width() / mapsize[0]) * n.x) + self.node_rad * 5,
                ((self.screen.get_height() / mapsize[1]) * n.y) + self.node_rad * 5
            )
            
            for direction in ["u", "d", "l", "r"]:
                if n.edges[direction]:
                    e = n.edges[direction]
                    self.draw_road_segment(n, e, direction, node_pos)
                    self.draw_cars_on_road(n, e, direction)
    
    def draw_road_segment(self, n, e, direction, node_pos):
        a, b = e.nodeP, e.nodeN
        nodeP_screen_x = ((self.screen.get_width() / mapsize[0]) * a.x) + self.node_rad * 5
        nodeP_screen_y = ((self.screen.get_height() / mapsize[1]) * a.y) + self.node_rad * 5
        nodeN_screen_x = ((self.screen.get_width() / mapsize[0]) * b.x) + self.node_rad * 5
        nodeN_screen_y = ((self.screen.get_height() / mapsize[1]) * b.y) + self.node_rad * 5
        
        prev_node_pos = pygame.Vector2(nodeP_screen_x if a != n else nodeN_screen_x,
                                     nodeP_screen_y if a != n else nodeN_screen_y)
        
        # Draw road
        pygame.draw.line(self.screen, "black", node_pos, prev_node_pos, width=self.road_width)
        
        # Draw yellow lines
        offset = int(self.road_width / 8)
        yWidth = int(self.road_width / 9)
        
        if direction in ["u", "d"]:
            for off in [offset, -offset]:
                pygame.draw.line(self.screen, "yellow",
                               pygame.Vector2(prev_node_pos.x + off, prev_node_pos.y),
                               pygame.Vector2(node_pos.x + off, node_pos.y),
                               width=yWidth)
        else:
            for off in [offset, -offset]:
                pygame.draw.line(self.screen, "yellow",
                               pygame.Vector2(prev_node_pos.x, prev_node_pos.y + off),
                               pygame.Vector2(node_pos.x, node_pos.y + off),
                               width=yWidth)
    
    def draw_cars_on_road(self, n, e, direction):
        nodeP_screen_x = ((self.screen.get_width() / mapsize[0]) * e.nodeP.x) + self.node_rad * 5
        nodeP_screen_y = ((self.screen.get_height() / mapsize[1]) * e.nodeP.y) + self.node_rad * 5
        nodeN_screen_x = ((self.screen.get_width() / mapsize[0]) * e.nodeN.x) + self.node_rad * 5
        nodeN_screen_y = ((self.screen.get_height() / mapsize[1]) * e.nodeN.y) + self.node_rad * 5
        
        delta_x = nodeN_screen_x - nodeP_screen_x
        delta_y = nodeN_screen_y - nodeP_screen_y
        offset = int(self.road_width / 8)
        
        # Draw cars in positive direction
        for c in e.carsP:
            fraction = c.position / e.length
            car_x = nodeP_screen_x + fraction * delta_x
            car_y = nodeP_screen_y + fraction * delta_y
            
            if direction in ["u", "d"]:
                car_rect = pygame.Rect(car_x + offset, car_y, self.carWidth, self.carHeight)
            else:
                car_rect = pygame.Rect(car_x, car_y + offset, self.carHeight, self.carWidth)
            
            pygame.draw.rect(self.screen, pygame.Color(c.Color), car_rect)
        
        # Draw cars in negative direction
        for c in e.carsN:
            fraction = c.position / e.length
            car_x = nodeN_screen_x - fraction * delta_x
            car_y = nodeN_screen_y - fraction * delta_y
            
            if direction in ["u", "d"]:
                car_rect = pygame.Rect(car_x - self.carWidth - offset, car_y, self.carWidth, self.carHeight)
            else:
                car_rect = pygame.Rect(car_x, car_y - self.carWidth - offset, self.carHeight, self.carWidth)
            
            pygame.draw.rect(self.screen, pygame.Color(c.Color), car_rect)
    
    def draw_interface(self):
        # Draw background
        s = max(self.screen.get_width(), self.screen.get_height())
        for i in range(0, s // self.bg.get_size()[0] + 1):
            for f in range(0, s // self.bg.get_size()[1] + 1):
                self.screen.blit(self.bg, (i * self.bg.get_size()[0], f * self.bg.get_size()[1]))
        
        # Draw roads and cars
        self.draw_roads_and_cars()
        
        # Draw nodes and traffic lights
        for n in nodes:
            node_pos = pygame.Vector2(
                ((self.screen.get_width() / mapsize[0]) * n.x) + self.node_rad * 5,
                ((self.screen.get_height() / mapsize[1]) * n.y) + self.node_rad * 5
            )
            node_color = "green" if n.lightud else "red"
            pygame.draw.circle(self.screen, node_color, node_pos, self.node_rad)
            
            # Draw car count
            t_surface = self.node_font.render(str(len(n.cars_in_intersection)), False, (11, 15, 106))
            self.screen.blit(t_surface, (node_pos.x - 0.5 * self.node_rad, node_pos.y - 1.5 * self.node_rad))
        
        # Draw statistics
        crashes_text = self.my_font.render(f'Crashes: {int(get_crashes()/2)}', False, (11, 15, 106))
        successes_text = self.my_font.render(f'Successes: {get_successes()}', False, (11, 15, 106))
        mode_text = self.info_font.render(f'Mode: {"AI" if self.using_model else "Random"}', False, (11, 15, 106))
        
        self.screen.blit(crashes_text, (0, 0))
        self.screen.blit(successes_text, (0, 50))
        self.screen.blit(mode_text, (0, 120))
    
    def run(self):
        while self.running:
            # Process game events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.using_model = not self.using_model if self.model is not None else False
            
            # Update simulation
            tick_all()
            
            # Update traffic lights based on timer
            self.time_elapsed_since_last_action += self.clock.get_time() / 1000
            if self.time_elapsed_since_last_action > self.update_interval:
                self.update_traffic_lights()
                self.time_elapsed_since_last_action = 0
            
            # Clear screen and draw everything
            self.screen.fill("white")
            self.draw_interface()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    visualizer = ModelVisualizer()
    visualizer.run()