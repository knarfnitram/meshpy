import numpy as np

class Node_Position_Tracker:
    """
    This object keeps track of the positions (or displacment) of nodes.
    It captures the positions of a node at certain times and can return the
    position sorted by ascending time.
    """
    def __init__(self):

        # list containing nodes
        self.nodes=[]

        # list containing the times
        self.times=[]

        # list containting the positions
        self.positions=[]

    """
    Add or update the position and time of a node
    """
    def add(self,node,time,position):

        if node in self.nodes:
            index=self.nodes.index(node)
            self.times[index]=np.append(self.times[index],time)
            self.positions[index]=np.vstack((self.positions[index],position))
        else:
            self.nodes.append(node)
            index= len(self.nodes) - 1
            self.times.append(time)
            self.positions.append(position)


        if np.linalg.norm(node.coordinates-self.nodes[index].coordinates)>1e-8:
            raise ValueError("You are trying to add values to the wrong node.")

        if int(time.shape[0]) - int(position.shape[0]) != 0:
            print(time.shape[0],position.shape[0])
            raise ValueError("You are trying to add times and positions with different lenghts")

        if int(position.shape[1])!=3:
            raise ValueError("Positions shape must always be 3.")


    """
    returns an array with ascending time and positions 
    """
    def get_sorted_by_time(self,node):
        if node in self.nodes:
            index=self.nodes.index(node)
            sorted_indices=np.argsort(self.times[index])
            return self.times[index][sorted_indices],self.positions[index][sorted_indices]
        else:
            raise ValueError("Couldn't find node with coordinates:",node.coordinates)
