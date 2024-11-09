import random, pickle

edges = []
nodes = []
cars = []

mapsize = [10,10]
car_breaking_range = (0.1,0.3)
traffic_light_range = .4
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


def path(start):
    # First, create a random target node (select a node other than the start node)
    random_target = None
    while random_target is None or random_target == start:
        random_target = random.choice(nodes)  # Randomly select a node, ensuring it's not the start node
    # Initialize BFS structures
    queue = [start]  # BFS queue
    came_from = {start: None}  # Dictionary to track the parent of each node
    
    # BFS loop
    while queue:
        current_node = queue.pop(0)
        
        # Check if we've reached the target node
        if current_node == random_target:
            break
        
        # Get all neighbors of the current node
        neighbors = get_neighbor(current_node)
        
        for neighbor, direction in neighbors:
            if neighbor not in came_from:  # If neighbor hasn't been visited
                queue.append(neighbor)
                came_from[neighbor] = current_node
    
    # Reconstruct the path from target to start by following parent nodes
    path_list = []
    current_node = random_target
    while current_node is not None:
        path_list.append(current_node)
        current_node = came_from[current_node]
    
    path_list.reverse()  # Reverse to get path from start to target
    return path_list





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
    ticked = False
    d_path = None

    accel = .005/60#add random later
    brake_accel = .1/60#add random later
    time_in_intersection = 0
    def pathfind(self):
        pass #self.path = list of nodes we want to get to
    
    def __init__(self, start):
        self.brake_dist = random.randint(int(car_breaking_range[0]*100),int(car_breaking_range[1]*100)+1)/100
        self.d_path = [start]
        for i in range(5):
            self.d_path.append(random.choice(get_neighbor(self.d_path[-1]))[0])
        del self.d_path[0]
    #def __init__(self, start):
    #    d_path = path(start)
    #    brake_dist = random.randint(int(car_breaking_range[0]*100),int(car_breaking_range[1]*100)+1)/100
    """
    def __init__(self, start):
        d_path = path(start)
        brake_dist = random.randint(int(car_breaking_range[0]*100),int(car_breaking_range[1]*100)+1)/100
    """
    def __init__(self):
        brake_dist = random.randint(int(car_breaking_range[0]*100),int(car_breaking_range[1]*100)+1)/100

    def __str__(self):
        return str([attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")])
        
    def crash(self,container):
        try:    
            container.nodeN.remove(self)
        except:
            pass
        try:
            container.nodeP.remove(self)
        except:
            pass
        try:
            container.cars_in_intersection.remove(self)
        except:
            pass
        del self

class edge():
    ud = False
    def get_length(self):
        self.length = ((self.nodeP.x-self.nodeN.x)**2+(self.nodeP.y-self.nodeN.y)**2)**0.5
    def __init__(self, nodeP, nodeN): #carsp = cars going in positive direction; carsn = cars going in negative direction
        self.nodeP,self.nodeN=nodeP,nodeN
        self.carsP,self.carsN=[],[]
        self.speed_limit = 0.5/60
        self.get_length()
        self.ud = bool(abs(nodeP.y-nodeN.y))
    
    
    def ctick(self):
        # Process both directions: positive and negative
        for cars, node in [(self.carsP, self.nodeN), (self.carsN, self.nodeP)]:
            for i in range(len(cars)):
                current_car = cars[i]
                should_brake = False
                
                # Check for car ahead
                if i < len(cars) - 1:  # If not the last car
                    next_car = cars[i + 1]
                    between = next_car.position - current_car.position
                    
                    # Check for collision
                    if between < 0:
                        current_car.crash(self)
                        next_car.crash(self)
                        continue
                    
                    # Check if too close to car ahead
                    if between <= current_car.brake_dist:
                        should_brake = True
                
                # Check stoplight
                if (node.lightud ^ self.ud) and (self.length-current_car.position)<=traffic_light_range:  # If light is red
                    should_brake = True
                #print(self.length-current_car.position, node.lightud, self.ud, should_brake)

                # Apply acceleration/deceleration
                if should_brake:
                    current_car.speed = max(0, current_car.speed - current_car.brake_accel)
                else:
                    current_car.speed = min(self.speed_limit, current_car.speed + current_car.accel)
                current_car.position += current_car.speed
                #print(current_car)
                
                        
                    

    
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
        
    #print(node)
for n in nodes: 
    n.lightud = bool(random.randint(0,1))


[print(n) for n in path(nodes[0])]
