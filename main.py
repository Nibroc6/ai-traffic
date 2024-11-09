import random


edges = []
nodes = []
cars = []

mapsize = [10,10]


class node():
    edges={"u":None,"d":None,"l":None,"r":None}
    lightud=False
    cars_in_intersection = []
	def __init__(self,pos): #pos is a list or tuple of length 2 (x,y)
        self.x,self.y=pos
        
    def move_car(self):
        pass

class car():
    speed = 0
    def pathfind(self):
        pass #self.path = list of nodes we want to get to
    
	def __init__(self,):
		self.pathfind()
        

class edge():
	def __init__(self, carsP, carsN, nodeP, nodeN): #carsp = cars going in positive direction; carsn = cars going in negative direction
        self.nodeP,self.nodeN=nodeP,nodeN
        self.carsP,self.carsN = carsP,carsN
        self.cars = cars
        
        
if __name__="__main__":
    for y in range(mapsize[0]):
        for x in range(mapsize[1]):
            if random.randint(0,2):
                nodes.append(node([x,y]))
    
    print(nodes)
    for node in nodes:
        pass
        #i
        #edges.append(
            