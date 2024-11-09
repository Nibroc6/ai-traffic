import pygame
import random
from main import *



# pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()
running = True
dt = 0




#node radius
if(screen.get_width() >= screen.get_height()):
    node_rad = (screen.get_width() / mapsize[0])/10
else:
    node_rad = (screen.get_height() / mapsize[1])/10


    

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
            a = e.nodeP
            b = e.nodeN
            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = 1)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = 1)
        if(n.edges["d"]):
            e = n.edges["d"]
            a = e.nodeP
            b = e.nodeN
            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = 1)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = 1)
        if(n.edges["l"]):
            e = n.edges["l"]
            a = e.nodeP
            b = e.nodeN
            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = 1)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = 1)
        if(n.edges["r"]):
            e = n.edges["r"]
            a = e.nodeP
            b = e.nodeN
            if(a != n):
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * a.x) + node_rad, ((screen.get_height() / mapsize[1]) * a.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = 1)
            else:
                prev_node_pos = pygame.Vector2(((screen.get_width() / mapsize[0]) * b.x) + node_rad, ((screen.get_height() / mapsize[1]) * b.y) + node_rad)
                pygame.draw.line(screen, "black", node_pos, prev_node_pos, width = 1)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()