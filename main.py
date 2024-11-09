import random, pickle


edges = []
nodes = []
cars = []

mapsize = [10,10]
inverse_directions = {"u":"d","d":"u","l":"r","r":"l"}

def node_by_pos(x,y):
    for n in nodes:
        if n.x == x and n.y == y:
            return n
    return False

def find_next_node(node, d):
    x,y = node.x,node.y
    if d in "ud":
        while y>=0 and y<mapsize[1]:
            if d == "u":
                y-=1
            else:
                y+=1
            if new_node := node_by_pos(x,y):
                return new_node
    if d in "lr":
        while x>=0 and x<mapsize[0]:  # Changed to use x coordinates
            if d == "l":
                x-=1  # Modified x instead of y
            else:
                x+=1  # Modified x instead of y
            if new_node := node_by_pos(x,y):
                return new_node
    return False
        


class node():
    def __init__(self,pos): #pos is a list or tuple of length 2 (x,y)
        self.lightud=False
        self.cars_in_intersection = []
        self.edges={"u":None,"d":None,"l":None,"r":None}
        self.x,self.y=pos
        
    def move_car(self):
        pass
        
    def __str__(self):
        return str(self.edges)+"\n"+str(self.lightud)+"\n"+str(self.cars_in_intersection)+"\n"+str(f"({self.x},{self.y})")

class car():
    speed = 0
    position = 0
    def pathfind(self):
        pass #self.path = list of nodes we want to get to
    
    def __init__(self,):
        self.pathfind()
        

class edge():
    
    def __init__(self, nodeP, nodeN): #carsp = cars going in positive direction; carsn = cars going in negative direction
        self.nodeP,self.nodeN=nodeP,nodeN
        self.cars = cars
        self.carsP,self.carsN=[],[]
        
    
#create nodes ---------------
for y in range(mapsize[0]):
    for x in range(mapsize[1]):
        if random.randint(0,2):
            nodes.append(node([x,y]))
#print(nodes)
#print(nodes[0].x,nodes[0].y,n:=next_node(nodes[0],"d"))
#if n: print(n.x,n.y)
#create edges ---------------
for n in range(len(nodes)):
    #print("Pre-edit: ",n,nodes[n])
    node=nodes[n]
    for direction in "udlr":
        print(node)
        print(direction)
        if node.edges[direction] == None:
            next_node = find_next_node(node,direction)
            #print(next_node)
            if next_node:
                new_edge = edge(node,next_node)
                edges.append(new_edge)
                node.edges[direction] = new_edge
                next_node.edges[inverse_directions[direction]] = new_edge
            else:
                node.edges[direction] = False
        
    print("Post-edit: ",node)
print(edges)
