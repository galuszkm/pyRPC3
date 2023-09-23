import copy
import matplotlib as mpl
from fatpack import find_reversals, find_rainflow_cycles
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from DamageEquivalentSignal_02 import DamageEquivalentSignal

class Channel_Class:

    def __init__(self, number:int=None, name:str='', units:str='', scale:float=None, dt:float=None):
    
        # RLD level metadata (same for channels acros all events):
        self.phisical_quantity = None
        self.location = None
        self.coordinate_system = None
        self.coordinate_system_attachment = None
        self.direction = None
        self.unit = None
        self.car_end = None
        self.car_side = None
        self.data_source = None

        self.database_attrs = [    
            'name',            
            'number',                       
            'phisical_quantity',
            'location',
            'coordinate_system',
            'coordinate_system_attachment',
            'direction',
            'unit',
            'car_end',
            'car_side',
            'data_source',]

        # Channel level metadata (specific to given event and channel):
        self.max_value = None
        self.min_value = None
        self.mean_value = None
        self.RSM_value = None        
        self.damage_slope_3 = None
        self.damage_slope_5 = None
        self.damage_slope_10 = None
        self.damage_slope_22 = None                      

        # Init props
        self.number = number
        self.name = name
        self.units = units
        self._scale_ = scale
        self.dt = dt
        self.parent_event = None
        
        # Initialize values list
        self.value:np.ndarray = np.array([])
        
        # Init rainflow related props
        self.RF_reversals = None 
        self.RF_reversals_ix = None    

        self.RF_cycles = None
        self.RF_residuals = None    

        self.RF_range = None
        self.RF_mean = None
        self.Range = None
        self.Damage = None
        self.Ncum = None
        self.Dcum = None
        self.Hist_xbin = None
        self.Hist_ybin = None
        self.RF_matrix = None

    def purge(self):       
        # Initialize values list
        self.value = np.array([])
        
        # Init rainflow related props
        self.RF_reversals = None 
        self.RF_reversals_ix = None    
        self.RF_cycles = None
        self.RF_residuals = None    
        self.RF_range = None
        self.RF_mean = None
        self.Range = None
        self.Damage = None
        self.Ncum = None
        self.Dcum = None
        self.Hist_xbin = None
        self.Hist_ybin = None
        self.RF_matrix = None
        self.cycles_repetitions = None

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

    def rainflow_counting(self, repeat:float=1., close_residuals:bool=False):

        # Here we want to make sure that the counting is alwyas done from scratch:
        self.RF_reversals = None 
        self.RF_reversals_ix = None    
        self.RF_cycles = None
        self.RF_residuals = None    
        self.RF_range = None
        self.RF_mean = None
        self.Range = None
        self.Damage = None
        self.Ncum = None
        self.Dcum = None
        self.Hist_xbin = None
        self.Hist_ybin = None
        self.RF_matrix = None    
        self.cycles_repetitions = None 

        # Find reversals (peaks and valleys) and indexes of reversals in signal
        self.RF_reversals, self.RF_reversals_ix = find_reversals(self.value, k=500)

        # Calculate closed cycles ( [ [peak1, valley1], [peak2, valley2], ... ] )
        # and residuals
        self.RF_cycles, self.RF_residuals = find_rainflow_cycles(self.RF_reversals)

        # Multiply closed cycles by number of repetitions
        # self.RF_cycles = np.repeat(self.RF_cycles, repeat, axis=0)
        
        if close_residuals:
            # Close residuals
            # closed_residuals = concatenate_reversals(self.RF_residuals, self.RF_residuals)
            # function concatenate_reversals does't work correctly
            # for that reason, reversals are concatinated in following way:
            closed_residuals = np.concatenate((self.RF_residuals, self.RF_residuals), axis=0).flatten()
            closed_residuals, _ = find_reversals(closed_residuals, k=500)

            # Count cycles of closed residuals in one repetition
            cycles_residue, _ = find_rainflow_cycles(closed_residuals)
                   
            # Add closed cycles to closed residual cycles
            if self.RF_cycles.size > 0:
                self.RF_cycles = np.concatenate((self.RF_cycles , cycles_residue))
            else:
                self.RF_cycles = cycles_residue
        
        # Find the rainflow ranges from the cycles
        self.RF_range = np.abs(self.RF_cycles[:, 1] - self.RF_cycles[:, 0])
        self.RF_mean = (self.RF_cycles[:, 1] + self.RF_cycles[:, 0])/2
        self.cycles_repetitions = np.full(self.RF_range.__len__(), repeat)

    def calculate_damage(self, slope:float, gate:float=0, plot_graphs:bool=False):
        
        #Check if rainflow counting was performed
        if not (self.RF_range.any() and self.RF_mean.any()):
            print('No counted cycles, please run a rain-flow counting first')

        if gate>0:
            # --------- Apply gate to range list ------------------
            RangeMax = self.RF_range.max()

            idx = np.where(self.RF_range >= (gate/100 * RangeMax))
            
            self.RF_range = self.RF_range[idx]
            self.RF_mean = self.RF_mean[idx]

        # Calculate potential damage of full signal
        self.damage_per_cycle = self.RF_range ** slope
        self.demage_per_cycle_multiplied = np.multiply(self.damage_per_cycle, self.cycles_repetitions)
        self.Damage = np.sum(self.demage_per_cycle_multiplied)



        if plot_graphs == True:
            # ------------------ Count unique cycles -------------------
            self.Range, self.Cycles = np.unique(self.RF_range, return_counts=True)

            # Flip arrays for Ncum and Dcum calcs
            self.Cycles = np.flipud(self.Cycles)
            self.Range = np.flipud(self.Range)
            
            # Calculate potential damage of each block
            damage = []
            for i in range(self.Range.shape[0]):
                damage.append( (self.Range[i] ** slope * self.Cycles[i]) / self.Damage * 100)
            damage = np.array(damage)
            
            # Calculate cumulative cycles and damage
            self.Ncum = []
            suma = 0
            for i in self.Cycles:
                suma += i
                self.Ncum.append(suma)
                
            self.Dcum = []
            suma = 0
            for i in damage:
                suma += i
                self.Dcum.append(suma)
            
            # Plot Graphs
            fig = plt.figure(figsize=(7, 10), dpi=90)
            fig.suptitle(f'Channel {self.number:d}: {self.name:s} \nSlope = {slope:d}      Total damage = {self.Damage:.3e}\n', fontsize=12)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            
            ax1.semilogx(self.Ncum, self.Range)
            ax1.set_xlabel('Cumulative cycles [-]')
            ax1.set_ylabel('Range [' + self.units + ']')
            ax1.grid()
            
            ax2.plot(self.Dcum, self.Range)
            ax2.set_xlabel('Percentage of total damage [%]')
            ax2.set_ylabel('Range [' + self.units + ']')
            ax2.grid()
            plt.show()

    def rainflow_Histogram(self, bins:int=5, hist_type:str="Min and Max", plot_hist:bool=False):
        
        if hist_type == 'Range and Mean':
        
            # Create 2d matrix with rainflow results
            data = np.array([self.RF_mean, self.RF_range]).T
            
            # Create x and y bins
            MeanMin = self.RF_mean.min()
            MeanMax = self.RF_mean.max()
            RangeMin = self.RF_range.min()
            RangeMax = self.RF_range.max()
            
            intX = abs(MeanMax - MeanMin)/bins
            rowbin = np.linspace(MeanMin - intX/2, MeanMax + intX/2, bins+2)
            intY = abs(RangeMax - RangeMin)/bins
            colbin = np.linspace(RangeMin - intY/2,  RangeMax + intY/2, bins+2)

        elif hist_type == "Min and Max":
            
            arrayMin = np.add(self.RF_mean, np.array(self.RF_range)/(-2))
            arrayMax = np.add(self.RF_mean, np.array(self.RF_range)/(2))
            
            # Create 2d matrix with rainflow results
            data = np.array([arrayMin, arrayMax]).T
            
            # Create x and y bins
            MINmin = arrayMin.min()
            MINmax = arrayMin.max()
            MAXmin = arrayMax.min()
            MAXmax = arrayMax.max()
            
            intX = abs(MINmax - MINmin)/bins
            rowbin = np.linspace(MINmin - intX/2, MINmax + intX/2, bins+2)
            intY = abs(MAXmax - MAXmin)/bins
            colbin = np.linspace(MAXmin - intY/2,  MAXmax + intY/2, bins+2)
        
        elif hist_type == "Range only":
        
            # Create 2d matrix with rainflow results
            data = np.array([[0.5]*len(self.RF_range), self.RF_range]).T
            
            # Create x and y bins
            RangeMin = self.RF_range.min()
            RangeMax = self.RF_range.max()
            
            rowbin = np.linspace(0, 1, 2)
            intY = abs(RangeMax - RangeMin)/bins
            colbin = np.linspace(RangeMin - intY/2,  RangeMax + intY/2, bins+2)
        
        # Calculate matrix
        #self.Hist_xbin, self.Hist_ybin, self.RF_matrix = find_rainflow_matrix(data, rowbin, colbin, return_bins=True)
        
        N = rowbin.size-1
        M = colbin.size-1
        mat = np.zeros((N, M), dtype=np.float32)
        
        # Find bin index of each of the cycles
        nrows = np.digitize(data[:, 0], rowbin)-1
        ncols = np.digitize(data[:, 1], colbin)-1
        
        # Include values on the rightmost edge in the last bin
        nrows[data[:, 0] == rowbin[-1]] = N - 1
        ncols[data[:, 1] == colbin[-1]] = M - 1

        # Build the rainflow matrix
        for nr, nc in zip(nrows, ncols):
            mat[nr, nc] += 1.
    
        self.Hist_xbin = rowbin
        self.Hist_ybin = colbin
        self.RF_matrix = mat
        
        # Plot Histogram
        if plot_hist == True:
        
            # Create grid
            X, Y = np.meshgrid(rowbin, colbin, indexing='ij')
            
            # define the colormap
            cmap = plt.cm.jet
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # force the first color entry to be white
            cmaplist[0] = (1, 1, 1, 1)
            
            # create the new map
            cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
            
            # define the color bins and normalize
            bounds = np.linspace(1, self.RF_matrix.max(), 9)
            bounds = np.insert(bounds, 0, 0)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            # Plot
            C = plt.pcolormesh(X, Y, self.RF_matrix, cmap=cmap, norm=norm)
            plt.colorbar(C)
            plt.title('Rainflow histogram')
            plt.xlabel('Mean [' + self.units + ']')
            plt.ylabel('Range [' + self.units + ']')
            plt.grid()
            plt.show()

    def equivalent_damage_block_signal(self, slope:int, block_no:int=5, min_cycles_no:int=2.5e5, repetitions:int=1):
    
        # Check if rainflow was performed:
        if (not self.RF_mean.any()) or (not self.RF_range.any()) or (not self.cycles_repetitions.any()):
            print("No rainflow data, please run rainflow counting first")
            return None
        elif len(self.RF_mean)!=len(self.RF_range) or len(self.RF_range)!=len(self.cycles_repetitions):
            print("Error! - Inconsistent rainflow data")
            return None

        # Prepare Rainflow input for Equivalent Block Script
        RF = []
        for i in range(len(self.RF_range)):
            item =  {'number': i,
                     'min_of_cycle': self.RF_mean[i] - self.RF_range[i]/2,
                     'max_of_cycle': self.RF_mean[i] + self.RF_range[i]/2,
                     'range': self.RF_range[i],
                     'cycle_repetitions': self.cycles_repetitions[i],
                    }
            item['damage_of_cycle'] =  item['cycle_repetitions'] * item['range']**slope
            item['percentage_cumulative_damage'] = item['damage_of_cycle']/self.Damage

            # Add item to Rainflow input list
            RF.append(item)
        
        # _temp_input = pd.DataFrame(RF)
        # print(f"The damge of the euivalent signal: {_temp_input['damage_of_cycle'].sum():.4e}")

        # Sort RF list by ascending range
        RF.sort(key=lambda i: i['range'])  
        
        # Second loop to calculate cumulative damage
        Dcum = 0
        for i in RF:
            Dcum += i['damage_of_cycle']
            i['cumul_damage'] = Dcum
        
        # Use Szymon's script to calculate equivalent signal
        result = DamageEquivalentSignal(RF, block_no, min_cycles_no, repetitions, slope, 1, None)
        
        # Store equivalent signals as Channel fields
        self.EqSignal = result[0]
        self.EqSignalRed = result[1]
        self.EqSignalMes = result[2]

    def Equivalent_Blocks2(self, block_no, min_cycles_no, slope):
        
        ### This method uses Rainflow counted on multiplicated signal
        ### But with huge number of repetitions (e.g. 100 000) it is 
        ### MUCH MUCH MUCH slower
        ### than first Equivalent_Blocks method.
        
        
        # First we need a list of unique pars (range, mean)
        # The following solution is much faster than numpy.unique()
        
        # Let's create a dict where keys are tuples (range, mean)
        # and values are number of repetitions of these pars
        y = dict()
        for i in range(len(self.RF_range)):
            
            # Define key as a tuple (range, mean)
            myKey = (self.RF_range[i], self.RF_mean[i])
            
            # If this key already exists - increase repetitions
            if myKey in y.keys():
                y[myKey] += 1
            
            # If not - create new item in dict
            else:
                y[myKey] = 1
        
        # Now let's create list of the following elements: [range, mean, repetitions]
        keys = list(y.keys())
        vals = list(y.values())
        x = [ [keys[i][0], keys[i][1], vals[i]] for i in range(len(keys)) ]
        
        # And sort it by ascending range
        x.sort(key=lambda i: i[0])
        
        
        # Prepare Rainflow input for Equivalent Block Script
        RF = []
        for i in range(len(x)):
            item =  {'number': i,
                     'min_of_cycle': x[i][1] - x[i][0]/2,
                     'max_of_cycle': x[i][1] + x[i][0]/2,
                     'range': x[i][0],
                     'cycle_repetitions': x[i][2],
                    }
            item['damage_of_cycle'] = item['range']**slope * item['cycle_repetitions']
            item['percentage_cumulative_damage'] = item['damage_of_cycle']/self.Damage
            
            # Add item to Rainflow input list
            RF.append(item)
            
        # Second loop to calculate cumulative damage
        Dcum = 0
        for i in RF:
            Dcum += i['damage_of_cycle']
            i['cumul_damage'] = Dcum
            
        # Use Szymon's script to calculate equivalent signal
        result = DamageEquivalentSignal(RF, block_no, min_cycles_no, 1, slope, 1, None)
        
        # Store equivalent signals as Channel fields
        self.EqSignal = result[0]
        self.EqSignalRed = result[1]
        self.EqSignalMes = result[2]
