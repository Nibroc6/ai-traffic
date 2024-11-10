import pygame
import random
from main import *

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()
running = True
dt = 0

# Background image
bgOriginal = pygame.image.load("grass.png")
bg = pygame.transform.scale(bgOriginal, (80, 80))

# Create initial cars on edges
#for e in edges:
#    length = e.length
#    if random.randint(0, 1):
#        # Cars going in positive direction (nodeP to nodeN)
#        c = car(e.nodeP, e.nodeN)
#        c.position = random.uniform(0, length)
#        e.carsP.insert(0, c)
#        
#        # Cars going in negative direction (nodeN to nodeP)
#        c = car(e.nodeN, e.nodeP)
#        c.position = random.uniform(0, length)
#        e.carsN.insert(0, c)

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

time_elapsed_since_last_action = 0


#amount of crashes
pygame.font.init()
my_font = pygame.font.SysFont('Arial Rounded MT Bold', 70)
#number of cars in node
node_font = pygame.font.SysFont('Comic Sans', 10)


while running:
    tick_all()
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    #update crashes
    text_surface = my_font.render('crashes: ' + str(int(get_crashes()/2)), False, (11, 15, 106))


    # Clear the screen
    screen.fill("white")

    # Draw background
    s = max(screen.get_width(), screen.get_height())
    for i in range(0, s // bg.get_size()[0] + 1):
        for f in range(0, s // bg.get_size()[1] + 1):
            screen.blit(bg, (i * bg.get_size()[0], f * bg.get_size()[1]))

    # Loop through nodes to render roads and cars
    for n in nodes:
        n.ctick()
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
                # Cars moving from nodeP to nodeN (positive direction)
                for c in e.carsP:
                    fraction = c.position / e.length
                    car_x = nodeP_screen_x + fraction * delta_x
                    car_y = nodeP_screen_y + fraction * delta_y

                    if direction in ["u", "d"]:
                        # Adjust positions for vertical roads
                        car_rect = pygame.Rect(car_x + offset, car_y, carWidth, carHeight)
                    else:
                        # Adjust positions for horizontal roads
                        car_rect = pygame.Rect(car_x, car_y + offset, carHeight, carWidth)
                    #pygame.Color(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
                    
                    pygame.draw.rect(screen, pygame.Color(c.Color), car_rect)

                # Cars moving from nodeN to nodeP (negative direction)
                for c in e.carsN:
                    fraction = c.position / e.length
                    car_x = nodeN_screen_x - fraction * delta_x
                    car_y = nodeN_screen_y - fraction * delta_y

                    if direction in ["u", "d"]:
                        # Adjust positions for vertical roads
                        car_rect = pygame.Rect(car_x - carWidth - offset, car_y, carWidth, carHeight)
                    else:
                        # Adjust positions for horizontal roads
                        car_rect = pygame.Rect(car_x, car_y - carWidth - offset, carHeight, carWidth)

                    pygame.draw.rect(screen, pygame.Color(c.Color), car_rect)
        

    # Draw traffic lights at nodes
    for n in nodes:
        node_pos = pygame.Vector2(
            ((screen.get_width() / mapsize[0]) * n.x) + node_rad * 5,
            ((screen.get_height() / mapsize[1]) * n.y) + node_rad * 5
        )
        node_color = "green" if n.lightud else "red"
        pygame.draw.circle(screen, node_color, node_pos, node_rad)
        #counter
        t_surface = node_font.render(str(len(n.cars_in_intersection)), False, (11, 15, 106))
        screen.blit(t_surface, (node_pos.x-node_rad,node_pos.y-node_rad))



    #draw crashes text
    screen.blit(text_surface, (0,0))

    # Update the display
    pygame.display.flip()

    # Limit FPS
    dt = clock.tick(240) / 1000
    #change the lights
    time_elapsed_since_last_action += dt
    if time_elapsed_since_last_action > .5:
        for n in nodes:
            n.lightud = not n.lightud
        time_elapsed_since_last_action = 0

pygame.quit()
