import random, pickle

edges = []
nodes = []
cars = []
mutiplier = 1
crashes = 0

with open(r"edges.obj", "wb") as node_file:
    pickle.dump(nodes, node_file)

with open(r"nodes.obj", "wb") as edge_file:
    pickle.dump(nodes, edge_file)
    
mapsize = [10,10]
car_breaking_range = (0.15,0.3)
crash_dist = 0.01
traffic_light_range = .4
max_time_in_intersection = 5
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
        
def transition(current_loc, car):
    if type(current_loc).__name__ == 'node':
        # Find the next node in the car's path
        try:
            current_idx = car.path.index(current_loc)
            if current_idx >= len(car.path) - 1:
                # Path is complete, remove car
                remove_item(current_loc.cars_in_intersection, car)
                return
                
            d_node = car.path[current_idx + 1]
            
            # First remove car from current location
            remove_item(current_loc.cars_in_intersection, car)
            
            # Reset car properties for new edge
            car.speed = 0
            car.position = 0
            
            # Then add to appropriate edge
            if d_node.x - current_loc.x > 0:
                current_loc.edges['r'].carsP.insert(0,car)  # Changed from insert(0)
            elif d_node.x - current_loc.x < 0:
                current_loc.edges['l'].carsN.insert(0,car)  # Changed from insert(0)
            elif d_node.y - current_loc.y < 0:
                current_loc.edges['u'].carsN.insert(0,car)  # Changed from insert(0)
            elif d_node.y - current_loc.y > 0:
                current_loc.edges['d'].carsP.insert(0,car)  # Changed from insert(0)
                
        except ValueError:
            print(f"Warning: Current node not found in car's path")
            return
            
    else:  # Edge
        try:
            next_idx = car.path.index(current_loc.nodeN)
            prev_idx = car.path.index(current_loc.nodeP)
            
            # Remove car from current edge first
            if next_idx > prev_idx:  # Car is moving toward nodeN
                remove_item(current_loc.carsP, car)
                current_loc.nodeN.cars_in_intersection.insert(0,car)  # Changed from insert(0)
            else:  # Car is moving toward nodeP
                remove_item(current_loc.carsN, car)
                current_loc.nodeP.cars_in_intersection.insert(0,car)  # Changed from insert(0)
                
            # Reset car properties for intersection
            car.speed = 0
            car.position = 0
            car.time_in_intersection = 0
            
        except ValueError:
            print(f"Warning: Node not found in car's path")
            return
        
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
    path_list.pop(0)
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
    
    def ctick(self):
        # Process cars in intersection
        i = 0
        while i < len(self.cars_in_intersection):
            car = self.cars_in_intersection[i]
            
            # Find the next edge the car will move to
            for j in range(len(car.path)-1):
                if car.path[j] == self:
                    next_node = car.path[j+1]
                    # Determine which edge we're heading to
                    if next_node.x - self.x > 0:
                        target_edge = self.edges['r']
                    elif next_node.x - self.x < 0:
                        target_edge = self.edges['l']
                    elif next_node.y - self.y < 0:
                        target_edge = self.edges['u']
                    elif next_node.y - self.y > 0:
                        target_edge = self.edges['d']
                    
                    # Adjust speed towards target edge's speed limit
                    target_speed = target_edge.speed_limit
                    if car.speed < target_speed:
                        car.speed = min(target_speed, car.speed + car.accel)
                    elif car.speed > target_speed:
                        car.speed = max(target_speed, car.speed - car.brake_accel)
                    break
            
            # Increment time in intersection
            car.time_in_intersection += 1
            
            # If car has been in intersection for 10 ticks, move it to next road
            if car.time_in_intersection >= max_time_in_intersection:
                car.time_in_intersection = 0  # Reset timer
                transition(self, car)  # This will move car to next road
                # Don't increment i since we removed a car
            else:
                i += 1

class car():
    speed = 0
    position = 0
    ticked = False
    path = None
    accel = .005/60
    brake_accel = 100
    time_in_intersection = 0
    
    def __init__(self, start, next_node=None):
        self.brake_dist = random.randint(int(car_breaking_range[0]*100),int(car_breaking_range[1]*100)+1)/100
        self.path = [start]
        
        if next_node:
            # If we're given a next node, make sure it's in the path
            self.path.append(next_node)
            # Then continue with random path from there
            current = next_node
            for i in range(4):  # Reduced by 1 since we already added next_node
                neighbors = get_neighbor(current)
                if not neighbors:
                    break
                next_choice = random.choice(neighbors)[0]
                if next_choice not in self.path:  # Avoid loops
                    self.path.append(next_choice)
                    current = next_choice
        else:
            # Original random path generation
            for i in range(5):
                neighbors = get_neighbor(self.path[-1])
                if not neighbors:
                    break
                next_choice = random.choice(neighbors)[0]
                if next_choice not in self.path:  # Avoid loops
                    self.path.append(next_choice)
    #def __init__(self, start):
    #    path = path(start)
    #    brake_dist = random.randint(int(car_breaking_range[0]*100),int(car_breaking_range[1]*100)+1)/100
    """
    def __init__(self, start):
        d_path = path(start)
        brake_dist = random.randint(int(car_breaking_range[0]*100),int(car_breaking_range[1]*100)+1)/100
    
    def __init__(self):
        brake_dist = random.randint(int(car_breaking_range[0]*100),int(car_breaking_range[1]*100)+1)/100
    """
    #def __str__(self):
    #    return str([self for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")])
        
    def crash(self,container):
        global crashes
        crashes += 1
        print(f"Car {self} crashed ({crashes} crashes so far)")
        try:    
            container.carsN.remove(self)
        except:
            print("removal error1")
            try:
                container.carsP.remove(self)
            except:
                print("removal error2")
                try:
                    container.cars_in_intersection.remove(self)
                except:
                    print("removal error3")
        del self

class edge():
    ud = False
    def get_length(self):
        self.length = ((self.nodeP.x-self.nodeN.x)**2+(self.nodeP.y-self.nodeN.y)**2)**0.5
    def __init__(self, nodeP, nodeN): #carsp = cars going in positive direction; carsn = cars going in negative direction
        self.nodeP,self.nodeN=nodeP,nodeN
        self.carsP,self.carsN=[],[]
        self.speed_limit = 0.2/60
        self.get_length()
        self.ud = bool(abs(nodeP.y-nodeN.y))
    
    
    def ctick(self):
        # Process both directions: positive and negative
        for cars, node in [(self.carsP, self.nodeN), (self.carsN, self.nodeP)]:
            i=0
            while i<len(cars):
                current_car = cars[i]
                should_brake = False
                
                # Check for car ahead
                if i < len(cars) - 1:  # If not the last car
                    next_car = cars[i + 1]
                    between = next_car.position - current_car.position
                    
                    # Check for collision
                    if between <= crash_dist:
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
                
                if current_car.position>=self.length:
                    transition(self, current_car)
                i+=1    

    
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
def tick_all():
    for m in range(mutiplier):
        for e in edges:
            e.ctick()
        for n in nodes: 
            n.ctick()