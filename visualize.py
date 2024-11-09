import pygame
import random
from main import *



# pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()
running = True
dt = 0


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
road_width = 3

#car width/height, I rly use these poorly so be warned if you try to use them
carHeight = 10
carWidth = 5

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")


    

    #loop through nodes
    #put a cirlce at node position
    for n in nodes:
        node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * n.x) + node_rad, ((screen.get_height() / mapsize[1]) * n.y) + node_rad)
        pygame.draw.circle(screen, "red", node_pos, node_rad)
        #go through 4 edges and go to their nodes. draw a line from n to that node
        if(n.edges["u"]):
            e = n.edges["u"]

            #calling backend tick moving car shit
            e.ctick()


            a = e.nodeP
            b = e.nodeN
            #distance x
            #two lists of cars
            #render cars
            for c in e.carsP:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                r = pygame.Rect(origin_pos[0]+(.5*road_width), origin_pos[1]+((screen.get_height() / mapsize[1])*c.position), carWidth, carHeight)
                pygame.draw.rect(screen, "blue", r)
            for c in e.carsN:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                r = pygame.Rect(origin_pos[0]-carWidth-(.5*road_width), origin_pos[1]-((screen.get_height() / mapsize[1])*c.position), carWidth, carHeight)
                pygame.draw.rect(screen, "green", r)

            #render road
            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
        if(n.edges["d"]):
            e = n.edges["d"]

            #calling backend tick moving car shit
            e.ctick()


            a = e.nodeP
            b = e.nodeN

            #distance x
            #two lists of cars
            #render cars
            for c in e.carsP:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                r = pygame.Rect(origin_pos[0]+(.5*road_width), origin_pos[1]+((screen.get_height() / mapsize[1])*c.position), carWidth, carHeight)
                pygame.draw.rect(screen, "blue", r)
            for c in e.carsN:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                r = pygame.Rect(origin_pos[0]-carWidth, origin_pos[1]-((screen.get_height() / mapsize[1])*c.position), carWidth, carHeight)
                pygame.draw.rect(screen, "green", r)


            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
        if(n.edges["l"]):
            e = n.edges["l"]

            #calling backend tick moving car shit
            e.ctick()


            a = e.nodeP
            b = e.nodeN

            #distance x
            #two lists of cars
            #render cars
            for c in e.carsP:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                r = pygame.Rect(origin_pos[0]+((screen.get_width() / mapsize[1])*c.position), origin_pos[1]+(.5*road_width), carHeight, carWidth)
                pygame.draw.rect(screen, "blue", r)
            for c in e.carsN:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                r = pygame.Rect(origin_pos[0]-((screen.get_width() / mapsize[1])*c.position), origin_pos[1]-carWidth, carHeight, carWidth)
                pygame.draw.rect(screen, "green", r)



            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
        if(n.edges["r"]):
            e = n.edges["r"]


            #calling backend tick moving car shit
            e.ctick()


            a = e.nodeP
            b = e.nodeN


            #distance x
            #two lists of cars
            #render cars
            for c in e.carsP:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                r = pygame.Rect(origin_pos[0]+((screen.get_width() / mapsize[1])*c.position), origin_pos[1]+(.5*road_width), carHeight, carWidth)
                pygame.draw.rect(screen, "blue", r)
            for c in e.carsN:
                origin_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                r = pygame.Rect(origin_pos[0]-((screen.get_width() / mapsize[1])*c.position), origin_pos[1]-carWidth, carHeight, carWidth)
                pygame.draw.rect(screen, "green", r)


            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = road_width)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()