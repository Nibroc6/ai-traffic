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
    accel = .005#add random later
    brake_accel = .01#add random later
    time_in_intersection = 0
    def pathfind(self):
        pass #self.path = list of nodes we want to get to
    
    def __init__(self):
        self.pathfind()
        brake_dist = random.randint(int(car_breaking_range[0]*100),int(car_breaking_range[1]*100)+1)/100
        
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
        self.speed_limit = 0.5
        self.get_length()
        ud = bool(abs(nodeP.y-nodeN.y))
    
    
    def tick(self):
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
                if (node.lightud ^ self.ud) and self.length-current_car.position<=traffic_light_range:  # If light is red
                    should_brake = True
                    
                # Apply acceleration/deceleration
                if should_brake:
                    current_car.speed = max(0, current_car.speed - current_car.brake_accel)
                else:
                    current_car.speed = min(self.speed_limit, current_car.speed + current_car.accel)
                    
                
                        
                    

    
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
        