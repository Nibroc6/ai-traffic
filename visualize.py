import pygame
import random
from main import *



# pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()
running = True
dt = 0


#background image
bgOriginal = pygame.image.load("grass.png")
bg = pygame.transform.scale(bgOriginal, (160, 160))

#make sum fuckin fcars
for e in edges:
    length = e.length
    if random.randint(0,1):
        # Cars going in positive direction (nodeP to nodeN)
        c = car(e.nodeP, e.nodeN)  # Pass both start and end nodes
        c.position = random.uniform(0, length)
        e.carsP.insert(0, c)
        
        # Cars going in negative direction (nodeN to nodeP)
        c = car(e.nodeN, e.nodeP)  # Pass both start and end nodes
        c.position = random.uniform(0, length)
        e.carsN.insert(0, c)





#node radius
if(screen.get_width() >= screen.get_height()):
    node_rad = (screen.get_width() / mapsize[0])/10
else:
    node_rad = (screen.get_height() / mapsize[1])/10

#road width
road_width = 9

#car width/height, I rly use these poorly so be warned if you try to use them
carHeight = 10
carWidth = 5


time_elapsed_since_last_action = 0
while running:
    tick_all()
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")
    #background image
    # screen.blit(bg, (0, 0))
    # screen.blit(bg, (bg.get_size()[0], 0))

    s = max(screen.get_width(), screen.get_height())
    for i in range(0, s//bg.get_size()[0] + 1):
        for f in range(0, s//bg.get_size()[1] + 1):
            screen.blit(bg, (i*bg.get_size()[0], f*bg.get_size()[1]))


    #loop through nodes
    #put a cirlce at node position
    for n in nodes:
        node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * n.x) + node_rad, ((screen.get_height() / mapsize[1]) * n.y) + node_rad)
        #yellow line offset
        offset = int(road_width/5)
        #yellow line width
        yWidth = int(road_width/9)
        #go through 4 edges and go to their nodes. draw a line from n to that node
        if(n.edges["u"]):
            e = n.edges["u"]

            #calling backend tick moving car shit


            a = e.nodeP
            b = e.nodeN

            #render road
            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            #yellow line right side
            lineStart = pygame.Vector2(prev_node_pos.x+offset, prev_node_pos.y)
            lineEnd = pygame.Vector2(node_pos.x+offset, node_pos.y)
            pygame.draw.line(screen, "yellow", lineStart, lineEnd, width = yWidth)
            #yellow line left side
            lineStart = pygame.Vector2(prev_node_pos.x-offset, prev_node_pos.y)
            lineEnd = pygame.Vector2(node_pos.x-offset, node_pos.y)
            pygame.draw.line(screen, "yellow", lineStart, lineEnd, width = yWidth)

            #distance x
            #two lists of cars
            #render cars
            for c in e.carsP:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                r = pygame.Rect(origin_pos[0]+offset, origin_pos[1]+((screen.get_height() / mapsize[1])*c.position), carWidth, carHeight)
                pygame.draw.rect(screen, "blue", r)
            for c in e.carsN:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                r = pygame.Rect(origin_pos[0]-carWidth-offset, origin_pos[1]-((screen.get_height() / mapsize[1])*c.position), carWidth, carHeight)
                pygame.draw.rect(screen, "green", r)
        if(n.edges["d"]):
            e = n.edges["d"]

            #calling backend tick moving car shit


            a = e.nodeP
            b = e.nodeN

            


            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            
            #distance x
            #two lists of cars
            #render cars
            for c in e.carsP:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                r = pygame.Rect(origin_pos[0]+offset, origin_pos[1]+((screen.get_height() / mapsize[1])*c.position), carWidth, carHeight)
                pygame.draw.rect(screen, "blue", r)
            for c in e.carsN:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                r = pygame.Rect(origin_pos[0]-carWidth-offset, origin_pos[1]-((screen.get_height() / mapsize[1])*c.position), carWidth, carHeight)
                pygame.draw.rect(screen, "green", r)
        if(n.edges["l"]):
            e = n.edges["l"]

            #calling backend tick moving car shit


            a = e.nodeP
            b = e.nodeN


            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            #yellow line right side
            lineStart = pygame.Vector2(prev_node_pos.x+offset, prev_node_pos.y)
            lineEnd = pygame.Vector2(node_pos.x+offset, node_pos.y)
            pygame.draw.line(screen, "yellow", lineStart, lineEnd, width = yWidth)
            #yellow line left side
            lineStart = pygame.Vector2(prev_node_pos.x-offset, prev_node_pos.y)
            lineEnd = pygame.Vector2(node_pos.x-offset, node_pos.y)
            pygame.draw.line(screen, "yellow", lineStart, lineEnd, width = yWidth)
        #distance x
        #two lists of cars
        #render cars
        for c in e.carsP:
            origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
            r = pygame.Rect(origin_pos[0]+((screen.get_width() / mapsize[1])*c.position), origin_pos[1]+offset, carHeight, carWidth)
            pygame.draw.rect(screen, "blue", r)
        for c in e.carsN:
            origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
            r = pygame.Rect(origin_pos[0]-((screen.get_width() / mapsize[1])*c.position), origin_pos[1]-carWidth-offset, carHeight, carWidth)
            pygame.draw.rect(screen, "green", r)
        if(n.edges["r"]):
            e = n.edges["r"]


            #calling backend tick moving car shit


            a = e.nodeP
            b = e.nodeN

            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            #distance x
            #two lists of cars
            #render cars
            for c in e.carsP:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                r = pygame.Rect(origin_pos[0]+((screen.get_width() / mapsize[1])*c.position), origin_pos[1]+offset, carHeight, carWidth)
                pygame.draw.rect(screen, "blue", r)
            for c in e.carsN:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                r = pygame.Rect(origin_pos[0]-((screen.get_width() / mapsize[1])*c.position), origin_pos[1]-carWidth-offset, carHeight, carWidth)
                pygame.draw.rect(screen, "green", r)
    for n in nodes:
        node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * n.x) + node_rad, ((screen.get_height() / mapsize[1]) * n.y) + node_rad)
        if n.lightud:
            node_color = "green"
        else:
            node_color = "red"
        pygame.draw.circle(screen, node_color, node_pos, node_rad)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(240) / 1000

    time_elapsed_since_last_action += dt
    if time_elapsed_since_last_action > 5:
        for n in nodes:
            n.lightud = not n.lightud
        time_elapsed_since_last_action = 0

pygame.quit()