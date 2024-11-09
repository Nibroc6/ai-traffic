from random import randint
from main import *

loc = (0, 0) # start node
des = (0, 0) # goal nodes
    
def path(n):
    goal_node = nodes[randint(len(nodes))]
    v_nodes = [n]
    path = [n]
    while not path[-1] == goal_node:
        #need a loop to determine next point from dists. likely a while
        dists = [ [abs((path[-1].x - n.x) + (path[-1].y - n.y)), n] for n in neighbors(path[-1])] # creates a list of distances between nodes.
        if len(dists) > 1 and not all(x in [n[1] for n in dists] for x in v_nodes):
            best = dists[dists.index([n for n in dists if n[0] == max([v[0] for v in dists])])]
            while not best[1] in v_nodes:
                dists.remove(best)
                best = dists[dists.index([n for n in dists if n[0] == max([v[0] for v in dists])])]
            path.append(best[1])
            v_nodes.append(best[1])
        else:
            path.pop()

#gets next steps
def neighbors(n):
    tmp = []
    for k in edges.keys():
        if not edges[k] == None:
            tmp.append([k, edges[k]]) #TODO: add the length of the path to the next node. second val should be the next node. Not the edge.
    return tmp