import struct
import sys
import math
import numpy as np
import os
from io import BufferedReader
from datetime import datetime
from .Channel import Channel_Class

def normalize_int16(array:np.ndarray):
    # Array bounds
    absmax_value = np.abs([array.max(), array.min()]).max()
    # Normalization factor
    factor = absmax_value/(2**15-1)
    # Normalize array
    if factor > 0:
        normalized_array = (array/factor).astype(np.int16)
    else:
        normalized_array = array.astype(np.int16)
        
    return normalized_array, factor
    
def writeRPC3(filename:str, dt:float, channels:list[Channel_Class]):
    __write_file__(filename, dt, channels)
     
class RPC3:
    def __init__(self, filename:str, debug:bool=False, extra_headers:dict={}, read_channels:list=None):
    
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
        
        # Set specific channels to read
        self.__channels_to_read__ = read_channels
        
        # Read file
        self.__read_file__()

    def info(self):
        """ Print summary of channels """  
        print('\n ===========================================================================================')
        sys.stdout.write("   %-15s %-30s %-15s %-15s %-15s\n" % ('Channel No', 'Name', 'Units', 'Min', 'Max'))
        print(' -------------------------------------------------------------------------------------------')
        for i in sorted(self.Channels, key = lambda x: x.number):
            sys.stdout.write(
                "     %-13s %-30s %-15s %-15s %-15s\n" % 
                (i.number, i.name, i.units, "%.3e" % i.getMin(), "%.3e" % i.getMax())
            )
        print(' ===========================================================================================\n')   
    
    def save(self, filename:str, dt:float, channels:list[Channel_Class]):
        __write_file__(filename, dt, channels)
    
    def getErrors(self) -> list[str]:
        return self.Errors
        
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
                __value__ = __value__.replace(b'\0', b'').decode('windows-1252').replace('\n', '').strip()
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
            _scale_ = 1.0
            if self.__data_type__ == 'SHORT_INTEGER':
                _scale_ = float(self.Headers['SCALE.CHAN_' + repr(channel + 1)])
                
            _channel_ = Channel_Class(
                channel + 1,  
                self.Headers['DESC.CHAN_' + repr(channel + 1)],
                self.Headers['UNITS.CHAN_' + repr(channel + 1)],
                _scale_,
                self.dt
            )
            self.Channels.append(_channel_)
                
            
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

        # Calculate total size needed for the channels array
        total_frames = point_per_frame * len([item for sublist in data_order for item in sublist])
        data_type_bytes = self.DATA_TYPES[self.__data_type__]['bytes']
        unpack_char = self.DATA_TYPES[self.__data_type__]['unpack_char']

        # Preallocate a NumPy array for each channel
        for channel in range(channels):
            self.Channels[channel].value = np.zeros(total_frames, dtype=np.float32)  # Adjust dtype as necessary
            
            # Default to all channels if specific_channels is None
            if self.__channels_to_read__ is None:
                specific_channels = range(channels)
            else:
                specific_channels = self.__channels_to_read__

        # Read and unpack in batches
        for i, frame_group in enumerate(data_order):
            for channel in range(channels):
                
                # Calculate the batch size for reading
                batch_size = len(frame_group) * point_per_frame * data_type_bytes
                
                # Fill the preallocated array with data if the channel is in specific_channels
                if channel in specific_channels:
                    batch_data = file_handle.read(batch_size)
                    # Unpack the entire batch at once
                    data_format = f'<{len(frame_group) * point_per_frame}{unpack_char}'
                    unpacked_data = struct.unpack(data_format, batch_data)
                    
                    # Fill the preallocated array with data
                    start_index = i * point_per_frame * len(frame_group)
                    end_index = start_index + len(frame_group) * point_per_frame
                    self.Channels[channel].value[start_index:end_index] = unpacked_data
                else:
                    # Skip the right number of bytes for non-specified channels
                    file_handle.seek(batch_size, 1)

        # Note: The code assumes each frame_group has the same number of frames, which might not be the case.
        # You may need to adjust logic for handling the last group or uneven groups.

        # Remove empty frame from channel values
        if removeLastFame:
            for channel in range(channels):
                if len(self.Channels[channel].value) > 0:
                    self.Channels[channel].value = self.Channels[channel].value[0:point_per_frame*frames]

        # Scale channel data
        for channel in range(channels):
            # Channel scale
            channel_scale = self.Channels[channel]._scale_

            # Scale data
            self.Channels[channel].value *= channel_scale

        #Remove empty channels
        indices_to_leave_set = set(self.__channels_to_read__)
        self.Channels = [obj for i, obj in enumerate(self.Channels) if i in indices_to_leave_set]

        return True

def __write_file__(filename:str, dt:float, channels:list[Channel_Class]):
    
    # Defaults
    PTS_PER_FRAME = 1024
    
    # Normalize channels
    __channels__ = [normalize_int16(c.value) for c in channels]
    
    # Determine max channel length, frames no and PTS_PER_GROUP
    __max_chan_len__ = max(len(i.value) for i in channels)
    FRAMES = math.ceil(__max_chan_len__/PTS_PER_FRAME)
    PTS_PER_GROUP = FRAMES * PTS_PER_FRAME
    
    # Collect channels header values - all must be string, scale allowed format is E8.6
    __chan_head__ = [
        [c.name, c.units, format(__channels__[idx][1], '8.6E')] 
        for idx, c in enumerate(channels)
    ]
    
    # Get encoded header
    __header__ = __write_header__(
        dt, 
        __chan_head__, 
        PTS_PER_FRAME, 
        FRAMES, 
        PTS_PER_GROUP,
    )
    
    # Get encoded data
    __data__ = __write_data__(
        [i[0] for i in __channels__],
        PTS_PER_GROUP,
    )
    
    # Write RPC3 file
    with open(filename, 'wb') as f:
        f.write(__header__)
        f.write(__data__)
    
def __write_header__(dt:float, chanData:list, PTS_PER_FRAME:int, FRAMES:int, PTS_PER_GROUP:int):
    
    # Current time
    ctime = datetime.now()
    
    # Header keys
    keys = [
        'FORMAT',
        'NUM_HEADER_BLOCKS',
        'NUM_PARAMS',
        'FILE_TYPE',
        'TIME_TYPE',
        'DELTA_T',
        'CHANNELS',
        'DATE',
        'REPEATS',
        'DATA_TYPE',
        'PTS_PER_FRAME',
        'PTS_PER_GROUP',
        'FRAMES',
    ]
    # Channel header keys
    channel_keys = [
        'DESC.CHAN_',
        'UNITS.CHAN_',
        'SCALE.CHAN_',
        'LOWER_LIMIT.CHAN_',
        'UPPER_LIMIT.CHAN_',
    ]
    
    # Header values
    values = [
        'BINARY',
        str(math.ceil((len(keys)+len(channel_keys)*len(chanData))/4)),  # Header block is 512 bytes: (32+96)*4
        str(len(keys) + len(channel_keys)*len(chanData)),
        'TIME_HISTORY',
        'RESPONSE',
        format(dt, '8.6E'),
        str(len(chanData)),
        f'{ctime.hour}:{ctime.minute}:{ctime.second} {ctime.day}-{ctime.month}-{ctime.year}',
        str(1),
        'SHORT_INTEGER',
        str(PTS_PER_FRAME),
        str(PTS_PER_GROUP),
        str(FRAMES),
    ]
    
    # Add channels headers keys and values
    for idx, cData in enumerate(chanData):
        keys += [i + str(idx+1) for i in channel_keys]
        values += [*cData, str(1), str(-1)]
    
    # Encode keys and values
    KEYS = [k.encode() for k in keys]
    VALUES = [v.encode() for v in values]
    
    # Write header bytes - adjust keys length to 32 bytes and values lenght to 96 bytes
    HEADER = b''
    for idx in range(len(KEYS)):
        HEADER += KEYS[idx].ljust(32, b'\x00') + VALUES[idx].ljust(96, b'\x00')
    
    # Required header length - based on NUM_HEADER_BLOCKS (each block is 512 bytes)
    __header_len__ = 512 * int(values[1])
    HEADER = HEADER.ljust(__header_len__, b'\x00')
    
    return(HEADER)

def __write_data__(data:list, PTS_PER_GROUP:int):
    
    # Init buffer
    DATA = b''
    
    # Iterate channels
    for d in data:
        # Fill channel to fit the group length
        if len(d) < PTS_PER_GROUP:
            d = np.pad(d, (0, PTS_PER_GROUP-len(d)), 'constant', constant_values=(d[-1]))
        
        # Add to bytes buffer
        DATA += d.tobytes()
        
    return DATA           