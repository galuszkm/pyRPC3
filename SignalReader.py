import struct
import sys
import copy
import math
from fatpack import find_reversals, find_rainflow_cycles, concatenate_reversals
import numpy as np
import os
from io import BufferedReader
import matplotlib as mpl
import matplotlib.pyplot as plt

class Channel_Class:

    def __init__(self, number:int, name:str, units:str, scale:float, dt:float, lowerlimit:float=None, upperlimit:float=None):
    
        # Init props
        self.Number = number
        self.Name = name
        self.Units = units
        self._scale = scale
        self.dt = dt
        self.LowerLimit = lowerlimit
        self.UpperLimit = upperlimit
        
        # Initialize values list
        self.value:np.ndarray = np.array([])
        
        # Init rainflow related props
        self.RF_range = 0
        self.RF_mean = 0
        self.Range = 0
        self.Cycles = 0
        self.Damage = 0
        self.Ncum = 0
        self.Dcum = 0
        self.Hist_xbin = 0
        self.Hist_ybin = 0
        self.RF_matrix = 0
             
    def getMax(self):
        return self.value.max()
               
    def getMin(self):
        return self.value.min()
              
    def rename(self, name:str):
        
        # Check if new name is type string
        if isinstance(name, str):
            self.Name = name
            return True
            
        else:
            print('\n ****** Channel new name must be type string! ******\n')
            return False     
            
    def renumber(self, number:int):
        
        # Check if new name is type string
        if isinstance(number, int):
            self.Number = number
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
        plt.title('Channel %i: %s' %(self.Number, self.Name))
        plt.xlabel('Time [s]')
        plt.ylabel(self.Units)
        plt.show()
        
    def rainflow_Counting(self, slope:float, gate:float, repet:float, plot_graphs:bool=False):
        
        # Change value list to numpy array
        y = np.array(self.value)
        
        # Find reversals (peaks and valleys) and indexes of reversals in signal
        reversals, reversals_ix = find_reversals(y, k=500)
        
        # Calculate closed cycles ( [ [peak1, valley1], [peak2, valley2], ... ] )
        # and residuals
        cycles, residue = find_rainflow_cycles(reversals)

        # Multiply closed cycles by number of repetitions
        cycles = np.repeat(cycles, repet, axis=0)
        
        # Close residuals
        closed_residuals = concatenate_reversals(residue, residue)
        
        # Count cycles of closed residuals in one repetition
        cycles_residue, _ = find_rainflow_cycles(closed_residuals)
        
        # Multiply closed residual cycles by number of repetitions
        cycles_residue = np.repeat(cycles_residue, repet, axis=0)
        
        # Add closed cycles to closed residual cycles
        if cycles.size > 0:
            cycles = np.concatenate((cycles , cycles_residue))
        else:
            cycles = cycles_residue
        
        # Find the rainflow ranges from the cycles
        self.RF_range = np.abs(cycles[:, 1] - cycles[:, 0])
        self.RF_mean = (cycles[:, 1] + cycles[:, 0])/2

        # Calculate potential damage of full signal
        self.Damage = np.sum(self.RF_range ** slope)

        # --------- Apply gate to range list ------------------
        RangeMax = self.RF_range.max()

        idx = np.where(self.RF_range >= (gate/100 * RangeMax))
        
        self.RF_range = self.RF_range[idx]
        self.RF_mean = self.RF_mean[idx]
        
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
        if plot_graphs == True:
            fig = plt.figure(figsize=(7, 10), dpi=90)
            fig.suptitle(f'Channel {self.Number:d}: {self.Name:s} \nSlope = {slope:d}      Total damage = {self.Damage:.3e}\n', fontsize=12)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            
            ax1.semilogx(self.Ncum, self.Range)
            ax1.set_xlabel('Cumulative cycles [-]')
            ax1.set_ylabel('Range [' + self.Units + ']')
            ax1.grid()
            
            ax2.plot(self.Dcum, self.Range)
            ax2.set_xlabel('Percentage of total damage [%]')
            ax2.set_ylabel('Range [' + self.Units + ']')
            ax2.grid()
            plt.show()

    def rainflow_Histogram(self, bins:int, hist_type:str, plot_hist:bool=False):
        
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
            plt.xlabel('Mean [' + self.Units + ']')
            plt.ylabel('Range [' + self.Units + ']')
            plt.grid()
            plt.show()
        
class RPC3:

    def __init__(self, filename:str, debug:bool=False, extra_headers:dict={}):
    
        # Init props
        self.Filename = filename
        self.debug = debug
        self.Headers = {}
        self.Channels:list[Channel_Class] = []
        self.Errors = []
        
        # Extra headers
        self.__extra_headers__ = {
          'INT_FULL_SCALE': int(2**15),
          'DATA_TYPE': 'SHORT_INTEGER',
          **extra_headers
        }
        
        # Signal timestep
        self.dt = 0
        
        # Allow for multiple data types from header DATA_TYPE.
        self.DATA_TYPES = {
            'FLOATING_POINT': {'unpack_char': 'f', 'bytes': 4},
            'SHORT_INTEGER': {'unpack_char': 'h', 'bytes': 2},
        }
        
        self.integer_standard_full_scale = 32768
        
        # Read file
        self.__read_file__()

    def __read_file__(self):
        
        # If file exists
        if os.path.isfile(self.Filename):
            
            # Open _file handle
            with open(self.Filename, 'rb') as file_handle:
                # Get _file size
                file_handle.seek(0, os.SEEK_END)
                self.__file_size__ = file_handle.tell()

                # Reposition to start of _file
                file_handle.seek(0, 0)
                
                if self.__read_header__(file_handle):
                    if self.__read_data__(file_handle):
                        return True
                    else:
                        return False
                else:
                    return False
        else:
            return False
             
    def __read_header__(self, file_handle:BufferedReader):
        
        def __header__():
            try:
                # Read header
                __head__, __value__ = struct.unpack('<32s96s', file_handle.read(128))
                __value__ = __value__.replace(b'\0', b'').decode('windows-1252').replace('\n', '')
                __head__ = __head__.replace(b'\0', b'').decode('windows-1252').replace('\n', '')
                return __head__, __value__
            except struct.error:
                self.Errors.append('Header of the file does not contain sufficient data to read 128 bytes')
                return None, None
            except UnicodeDecodeError:
                self.Errors.append('Header of the file could not be decoded properly')
                return None, None
                
        # Read the first position fixed headers
        for i in range(3):
            head_name, head_value = __header__()
            if head_name not in ['FORMAT', 'NUM_HEADER_BLOCKS', 'NUM_PARAMS']:
                self.Errors.append('Header of the file does not contain required fields')
                return False

            if head_name in ['NUM_HEADER_BLOCKS', 'NUM_PARAMS']:
                self.Headers[head_name] = int(head_value)
            else:
                self.Headers[head_name] = head_value
                
            # DEBUG
            if self.debug:
                print(f'\t {head_name:18s}: {head_value}')
            
        # Check if _file contains data
        if not self.Headers['NUM_PARAMS'] > 3:
            self.Errors.append(' No data in file')
            return False
            
        # Read all remaining headers
        for channel in range(3, self.Headers['NUM_PARAMS']):
            head_name, head_value = __header__()
            # Stored in blocks of 4 (512 bytes divided into 128 byte chunks), hence at the end empty headers can appear
            if head_name != None and len(head_name) != 0:
                self.Headers[head_name] = head_value
                if self.debug:
                    print(f"\t\t {head_name:32s}  -- {head_value}")
                    
        # Set current position in _file
        self.header_end = file_handle.tell()

        # Add additional headers
        for header_name, head_value in self.__extra_headers__.items():
            if header_name not in self.Headers:
                if self.debug: 
                    print(f' Adding extra header\n\t{header_name} - {head_value}')
                self.Headers[header_name] = head_value
            else:
                if self.debug:
                    print(f' WARNING: Extra header already defined in RPC file, skipping\n\t {header_name} - {head_value}')
                    
        # Convert values to correct types
        try:
            self.Headers['NUM_HEADER_BLOCKS'] = int(self.Headers['NUM_HEADER_BLOCKS'])
            self.Headers['CHANNELS'] = int(self.Headers['CHANNELS'])
            self.Headers['DELTA_T'] = float(self.Headers['DELTA_T'])
            self.Headers['PTS_PER_FRAME'] = int(self.Headers['PTS_PER_FRAME'])
            self.Headers['PTS_PER_GROUP'] = int(self.Headers['PTS_PER_GROUP'])
            self.Headers['FRAMES'] = int(self.Headers['FRAMES'])
            self.Headers['INT_FULL_SCALE'] = int(self.Headers['INT_FULL_SCALE'])
            self.__data_type__ = self.Headers['DATA_TYPE']
            self.dt = self.Headers['DELTA_T']
            
        except KeyError as expected_header:
            self.Errors.append(f'A mandatory header is missing: {expected_header}')
            return False
        
        # Structure channel data structure
        for channel in range(int(self.Headers['CHANNELS'])):
            
            # If scale header not included - e.g. for DATA_TYPE = FLOATING_POINT
            _scale = 1.0
            if self.__data_type__ == 'SHORT_INTEGER':
                _scale = float(self.Headers['SCALE.CHAN_' + repr(channel + 1)])
                
            _channel = Channel_Class(
                channel + 1,  
                self.Headers['DESC.CHAN_' + repr(channel + 1)],
                self.Headers['UNITS.CHAN_' + repr(channel + 1)],
                _scale,
                self.dt,
                lowerlimit = float(self.Headers['LOWER_LIMIT.CHAN_' + repr(channel + 1)]),
                upperlimit = float(self.Headers['UPPER_LIMIT.CHAN_' + repr(channel + 1)])
            )
            self.Channels.append(_channel)
                
            
        return True
    
    def __read_data__(self, file_handle:BufferedReader):

        channels = self.Headers['CHANNELS']
        point_per_frame = self.Headers['PTS_PER_FRAME']
        point_per_group = self.Headers['PTS_PER_GROUP']
        frames = self.Headers['FRAMES']

        # Read after end of header which occurs at 512 bytes times number of header blocks
        file_handle.seek(self.Headers['NUM_HEADER_BLOCKS'] * 512, 0)

        # Recreate structure of demultiplexed data
        frames_per_group = int((point_per_group / point_per_frame))
        number_of_groups = int(math.ceil(frames / frames_per_group))
        data_order = list()
        frame_no = 1
        removeLastFame = False
        
        for i in range(number_of_groups):
            if frame_no > frames:
                removeLastFame = True
            temp = list()
            for j in range(frames_per_group):
                if frame_no > frames:
                    removeLastFame = True
                temp.append(frame_no)
                frame_no += 1
            data_order.append(temp)
        del temp, frame_no
        
        if self.debug:
            print(' Data structure summary:'
                  f'\n\tChannels to read:  {channels}'
                  f'\n\tPoints per frame:  {point_per_frame}'
                  f'\n\tPoints per group:  {point_per_group}'
                  f'\n\tNumber of frames:  {frames}'
                  f'\n\tNumber of groups:  {number_of_groups}'
                  f"\n\tHeader end at:     {self.Headers['NUM_HEADER_BLOCKS'] * 512} bytes"
                  f"\n\tFile end at:       {self.__file_size__}"
                  f"\n\tBytes to read:     {self.__file_size__ - self.Headers['NUM_HEADER_BLOCKS'] * 512}")

            print(f' Frame grouping array:\n {data_order}')
            print(f" Binary decoding settings: <{point_per_frame}{self.DATA_TYPES[self.__data_type__]['unpack_char']}, "
                  f"{point_per_frame * self.DATA_TYPES[self.__data_type__]['bytes']}, "
                  f" Bytes per data value: {self.DATA_TYPES[self.__data_type__]['bytes']}")

        # Check that data type matches file size
        actual_data_size = self.__file_size__ - self.Headers['NUM_HEADER_BLOCKS'] * 512
        expected_data_size = point_per_frame * self.DATA_TYPES[self.__data_type__]['bytes'] * \
            frames_per_group * number_of_groups * channels

        if actual_data_size != expected_data_size:
            if self.debug:
                print(' ERROR: DATA_TYPE problem - Data cant be decoded correctly'
                    f'\n\tActual data size in bytes:   {actual_data_size}'
                    f'\n\tExpected data size in bytes: {expected_data_size}'
                    f'\n\tVerify that {self.__data_type__} is correct ')
            self.Errors.append('DATA_TYPE error')
            return False

        for frame_group in data_order:
            for channel in range(channels):
                for frame in frame_group:
                    data = struct.unpack(
                        f'<{point_per_frame}' + self.DATA_TYPES[self.__data_type__]['unpack_char'],
                        file_handle.read(point_per_frame * self.DATA_TYPES[self.__data_type__]['bytes'])
                    )

                    # Concatenate channel value array
                    self.Channels[channel].value = np.concatenate((self.Channels[channel].value, np.array(data)), axis=0)

        # Remove empty frame from channel values
        if removeLastFame:
            for channel in range(channels):
                self.Channels[channel].value = self.Channels[channel].value[0:point_per_frame*frames]

        # Scale channel data
        for channel in range(channels):
            # Channel scale
            channel_scale = self.Channels[channel]._scale
            # Standard integer full scale
            int_standard_full_scale = self.integer_standard_full_scale
            # RPC integer full scale
            int_rpc_full_scale = self.Headers['INT_FULL_SCALE']

            # Compute scale factor
            scale_factor = int_rpc_full_scale / int_standard_full_scale * channel_scale

            # Scale data
            self.Channels[channel].value *= scale_factor
            
        return True

    def info(self):
    
        print('\n ===========================================================================================')
        sys.stdout.write("   %-15s %-30s %-15s %-15s %-15s\n" % ('Channel No', 'Name', 'Units', 'Min', 'Max'))
        print(' -------------------------------------------------------------------------------------------')
        for i in sorted(self.Channels, key = lambda x: x.Number):
            sys.stdout.write("     %-13s %-30s %-15s %-15s %-15s\n" % (i.Number, i.Name, i.Units, "%.3e" % i.getMin(), "%.3e" % i.getMax()))
        print(' ===========================================================================================\n')    
