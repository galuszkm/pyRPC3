import copy
import numpy as np
import matplotlib.pyplot as plt

class Channel_Class:

    def __init__(self, number:int=None, name:str='', units:str='', scale:float=None, dt:float=None):

        # Init props
        self.number = number
        self.name = name
        self.units = units
        self._scale_ = scale
        self.dt = dt
        
        # Initialize values list
        self.value:np.ndarray = np.array([])

    def getMax(self):
        return self.value.max()

    def getMin(self):
        return self.value.min()

    def rename(self, name:str):
        
        # Check if new name is type string
        if isinstance(name, str):
            self.name = name
            return True
            
        else:
            print('\n ****** Channel new name must be type string! ******\n')
            return False     

    def renumber(self, number:int):
        
        # Check if new name is type string
        if isinstance(number, int):
            self.number = number
            return True
            
        else:
            print('\n ****** Channel new number must be type int! ******\n')
            return False

    def scale(self, factor:float):
        
        # Check if factor is numeric
        if isinstance(factor, int) or isinstance(factor, float):  
            self.value *= factor
            
        else:
            print('\n ****** Scale factor must be type int or float! ******\n')
            return False

    def copy(self):
        return copy.deepcopy(self)

    def plot(self, linewidth:float=1):
        time = np.array([i*self.dt for i in range(len(self.value))])
        plt.plot(time, self.value, linewidth=linewidth)
        plt.grid(True)
        plt.xlim(time.min(), time.max())
        plt.ylim(self.getMin(), self.getMax())
        plt.title('Channel %i: %s' %(self.number, self.name))
        plt.xlabel('Time [s]')
        plt.ylabel(self.units)
        plt.show()