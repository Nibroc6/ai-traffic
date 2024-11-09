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

def remove_item(l, i):
    try:
        l.remove(i)
    except  ValueError:
        return -1

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
        
def transition(current_loc, car): #cloc can be node or edge
    d_node = None
    if type(current_loc) is node:
        for i in range(len(car.path)-1):
            if car.path[i] == current_loc:
                d_node = car.path[i+1]
        if d_node:
            if d_node.x - current_loc.x > 0:
                current_loc.edges['r'].append(car)
            elif d_node.x - current_loc.x < 0:
                current_loc.edges['l'].append(car)
            elif d_node.y - current_loc.y < 0:
                current_loc.edges['u'].append(car)
            elif d_node.y - current_loc.y > 0:
                current_loc.edges['d'].append(car)
        remove_item(current_loc.cars_in_intersection, car)
    else:
        if car.path.index(current_loc.nodeP) > car.path.index(current_loc.nodeN) and not current_loc.nodeP == current_loc:
            current_loc.nodeP.append(car)
            remove_item(current_loc.nodeN, car)
        elif car.path.index(current_loc.nodeN) > car.path.index(current_loc.nodeP) and not current_loc.nodeN == current_loc:
            current_loc.nodeN.append(car)
            remove_item(current_loc.nodeP, car)
        else:
            remove_item(current_loc.nodeP, car)   
            remove_item(current_loc.nodeN, car)
        
def get_neighbor (node):
    #returns list of tuples. form [(node, dir), (node,dir)]
    out = []

    out.append((find_next_node(node, 'u'), 'u'))
    out.append((find_next_node(node, 'r'), 'r'))
    out.append ((find_next_node(node, 'd'), 'd'))
    out.append((find_next_node(node, 'l'), 'l'))
    
    result = []
    for i in range (len(out)):
        if not( out[i][0] == False or out[i][0] == None):
            result.append(out[i])
    return result


def path(n):
    goal_node = nodes[random.randint(0, len(nodes)-1)]
    
    v_nodes = [n]
    path = [n]
    print('goal:', goal_node)
    
    while path[-1] != goal_node:
        neighbors = get_neighbor(path[-1])
        if goal_node in [n[0] for n in neighbors]:
            path.append(goal_node)
            return path
        
        for n in neighbors:
            if n[0] in v_nodes:
                continue
            else:
                path.append()
    
    return path




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

    def __str__(self):
        return str(self.edges)+"\n"+str(self.lightud)+"\n"+str(self.cars_in_intersection)+"\n"+str(f"({self.x},{self.y})")

class car():
    speed = 0
    position = 0
    ticked = False
    d_path = None
    def pathfind(self):
        pass #self.path = list of nodes we want to get to
    
    def __init__(self, start):
        d_path = path(start)
        
    def __str__(self):
        return f"pos: {position} | desired path: {d_path}"

class edge():
    
    def __init__(self, nodeP, nodeN): #carsp = cars going in positive direction; carsn = cars going in negative direction
        self.nodeP,self.nodeN=nodeP,nodeN
        self.cars = cars
        self.carsP,self.carsN=[],[]
        
    def tick(self):
        for i in range(len(carsP)):
            if len(carsP)<i+1:
                if abs(carsP[i+1].position-carsP[i].position):
                    pass
        

    
#create nodes ---------------
for y in range(mapsize[0]):
    for x in range(mapsize[1]):
        if random.randint(0,2):
            nodes.append(node([x,y]))
#print(nodes)
#print(nodes[0].x,nodes[0].y,n:=next_node(nodes[0],"d"))
#if n: print(n.x,n.y)
#create edges ---------------

test = car(nodes[-1])
#[print(v) for v in get_neighbor(nodes[-1])]