import struct
import sys
import math
import numpy as np
import os
from io import BufferedReader
from .Channel import Channel
from .writter import write_rpc3

# Global constants
DATA_TYPES = {
    'FLOATING_POINT': {'unpack_char': 'f', 'bytes': 4},
    'SHORT_INTEGER': {'unpack_char': 'h', 'bytes': 2},
}

class RPC3:
    """
    Class to handle RPC3 file reading and writing.
    """

    def __init__(self, filename: str, debug: bool = False, extra_headers: dict = {}, read_channels: list = None) -> None:
        """
        Initialize an RPC3 instance.

        Args:
            filename (str): Path to the RPC3 file.
            debug (bool, optional): Enable debug output. Defaults to False.
            extra_headers (dict, optional): Additional header key-value pairs.
            read_channels (list, optional): Specific channel indices to read.
        """
        self.filename = filename
        self.debug = debug
        self.headers = {}
        self.channels: list[Channel] = []
        self.errors = []

        # Extra headers to add if not defined in file
        self._extra_headers = {
            'INT_FULL_SCALE': int(2**15),
            'DATA_TYPE': 'SHORT_INTEGER',
            **extra_headers
        }

        # Signal timestep
        self.dt = 0

        # Set specific channels to read
        self._channels_to_read = read_channels

        # Read file
        self._read_file()

    def info(self):
        """Print summary of channels."""
        print('\n' + '=' * 90)
        sys.stdout.write("{:<15s} {:<30s} {:<15s} {:<15s} {:<15s}\n".format(
            'Channel No', 'Name', 'Units', 'Min', 'Max'))
        print('-' * 90)
        for ch in sorted(self.channels, key=lambda x: x.number):
            sys.stdout.write(
                "{:<15s} {:<30s} {:<15s} {:<15.3e} {:<15.3e}\n".format(
                    str(ch.number), ch.name, ch.units, ch.get_min(), ch.get_max()
                )
            )
        print('=' * 90 + '\n')

    def save(self, filename: str, exclude_channels: list = None):
        """
        Save the RPC3 file using self.dt and self.channels, excluding channels that match
        any entry in exclude_channels (by channel number or channel name).

        Args:
            filename (str): Output file name.
            exclude_channels (list, optional): List of channel numbers or names to exclude.
        """
        if exclude_channels is None:
            channels_to_write = self.channels
        else:
            def is_excluded(ch: Channel) -> bool:
                for excl in exclude_channels:
                    if isinstance(excl, int) and ch.number == excl:
                        return True
                    elif isinstance(excl, str) and ch.name == excl:
                        return True
                return False

            channels_to_write = [ch for ch in self.channels if not is_excluded(ch)]

        write_rpc3(filename, self.dt, channels_to_write)

    def get_errors(self) -> list[str]:
        """
        Get a list of error messages.

        Returns:
            list[str]: Errors encountered.
        """
        return self.errors

    def _read_file(self):
        """
        Read the RPC3 file.

        Returns:
            bool: True if reading was successful, False otherwise.
        """
        if os.path.isfile(self.filename):
            with open(self.filename, 'rb') as file_handle:
                # Get file size
                file_handle.seek(0, os.SEEK_END)
                self._file_size = file_handle.tell()
                file_handle.seek(0, 0)

                if self._read_header(file_handle):
                    if self._read_data(file_handle):
                        return True
                    else:
                        return False
                else:
                    return False
        else:
            self.errors.append(f"File not found: {self.filename}")
            return False

    def _read_header(self, file_handle: BufferedReader):
        """
        Read the header from the file.

        Args:
            file_handle (BufferedReader): Open file handle.

        Returns:
            bool: True if header is read successfully, False otherwise.
        """

        def _read_header_entry():
            try:
                head, value = struct.unpack('<32s96s', file_handle.read(128))
                value = value.replace(b'\0', b'').decode('windows-1252').replace('\n', '').strip()
                head = head.replace(b'\0', b'').decode('windows-1252').replace('\n', '')
                return head, value
            except struct.error:
                self.errors.append('Header does not contain sufficient data (128 bytes expected).')
                return None, None
            except UnicodeDecodeError:
                self.errors.append('Header could not be decoded properly.')
                return None, None

        # Read the first fixed headers
        for i in range(3):
            head_name, head_value = _read_header_entry()
            if head_name not in ['FORMAT', 'NUM_HEADER_BLOCKS', 'NUM_PARAMS']:
                self.errors.append('Header does not contain required fields.')
                return False

            if head_name in ['NUM_HEADER_BLOCKS', 'NUM_PARAMS']:
                self.headers[head_name] = int(head_value)
            else:
                self.headers[head_name] = head_value

            if self.debug:
                print(f'\t{head_name:18s}: {head_value}')

        # Check if file contains data
        if not self.headers['NUM_PARAMS'] > 3:
            self.errors.append('No data in file.')
            return False

        # Read remaining headers
        for channel in range(3, self.headers['NUM_PARAMS']):
            head_name, head_value = _read_header_entry()
            if head_name is not None and len(head_name) != 0:
                self.headers[head_name] = head_value
                if self.debug:
                    print(f"\t\t{head_name:32s} -- {head_value}")

        self.header_end = file_handle.tell()

        # Add additional headers if missing
        for header_name, head_value in self._extra_headers.items():
            if header_name not in self.headers:
                if self.debug:
                    print(f' Adding extra header: {header_name} - {head_value}')
                self.headers[header_name] = head_value
            else:
                if self.debug:
                    print(f' WARNING: Extra header already defined in RPC file, skipping: {header_name} - {head_value}')

        # Convert header values to proper types
        try:
            self.headers['NUM_HEADER_BLOCKS'] = int(self.headers['NUM_HEADER_BLOCKS'])
            self.headers['CHANNELS'] = int(self.headers['CHANNELS'])
            self.headers['DELTA_T'] = float(self.headers['DELTA_T'])
            self.headers['PTS_PER_FRAME'] = int(self.headers['PTS_PER_FRAME'])
            self.headers['PTS_PER_GROUP'] = int(self.headers['PTS_PER_GROUP'])
            self.headers['FRAMES'] = int(self.headers['FRAMES'])
            self.headers['INT_FULL_SCALE'] = int(self.headers['INT_FULL_SCALE'])
            self._data_type = self.headers['DATA_TYPE']
            self.dt = self.headers['DELTA_T']
        except KeyError as expected_header:
            self.errors.append(f'A mandatory header is missing: {expected_header}')
            return False

        # Create channel objects
        for channel in range(self.headers['CHANNELS']):
            # Use scale header if available; default to 1.0 otherwise.
            scale = 1.0
            if self._data_type == 'SHORT_INTEGER':
                scale = float(self.headers.get('SCALE.CHAN_' + repr(channel + 1), 1.0))
            ch = Channel(
                channel + 1,
                self.headers.get('DESC.CHAN_' + repr(channel + 1), f'Channel {channel+1}'),
                self.headers.get('UNITS.CHAN_' + repr(channel + 1), ''),
                self.dt,
                scale,
            )
            self.channels.append(ch)

        return True

    def _read_data(self, file_handle: BufferedReader):
        """
        Read the data portion of the RPC3 file.

        Args:
            file_handle (BufferedReader): Open file handle positioned after the header.

        Returns:
            bool: True if data is read successfully, False otherwise.
        """
        channels = self.headers['CHANNELS']
        pts_per_frame = self.headers['PTS_PER_FRAME']
        pts_per_group = self.headers['PTS_PER_GROUP']
        frames = self.headers['FRAMES']

        # Seek to the data section (after header blocks)
        file_handle.seek(self.headers['NUM_HEADER_BLOCKS'] * 512, 0)

        frames_per_group = int(pts_per_group / pts_per_frame)
        number_of_groups = int(math.ceil(frames / frames_per_group))
        data_order = []
        frame_no = 1
        remove_last_frame = False

        for i in range(number_of_groups):
            temp = []
            for j in range(frames_per_group):
                if frame_no > frames:
                    remove_last_frame = True
                    break
                temp.append(frame_no)
                frame_no += 1
            data_order.append(temp)

        if self.debug:
            print('Data structure summary:'
                  f'\n\tChannels to read: {channels}'
                  f'\n\tPoints per frame: {pts_per_frame}'
                  f'\n\tPoints per group: {pts_per_group}'
                  f'\n\tNumber of frames: {frames}'
                  f'\n\tNumber of groups: {number_of_groups}'
                  f"\n\tHeader end at: {self.headers['NUM_HEADER_BLOCKS'] * 512} bytes"
                  f"\n\tFile end at: {self._file_size}"
                  f"\n\tBytes to read: {self._file_size - self.headers['NUM_HEADER_BLOCKS'] * 512}")

            print(f'Frame grouping array:\n{data_order}')
            print(f"Binary decoding settings: <{pts_per_frame}{DATA_TYPES[self._data_type]['unpack_char']}, "
                  f"{pts_per_frame * DATA_TYPES[self._data_type]['bytes']} bytes per frame, "
                  f"Bytes per data value: {DATA_TYPES[self._data_type]['bytes']}")

        actual_data_size = self._file_size - self.headers['NUM_HEADER_BLOCKS'] * 512
        expected_data_size = pts_per_frame * DATA_TYPES[self._data_type]['bytes'] * \
                             frames_per_group * number_of_groups * channels

        if actual_data_size != expected_data_size:
            if self.debug:
                print('ERROR: DATA_TYPE problem - Data cannot be decoded correctly'
                      f'\n\tActual data size in bytes: {actual_data_size}'
                      f'\n\tExpected data size in bytes: {expected_data_size}'
                      f'\n\tVerify that {self._data_type} is correct')
            self.errors.append('DATA_TYPE error')
            return False

        total_frames = pts_per_frame * sum(len(group) for group in data_order)
        data_type_bytes = DATA_TYPES[self._data_type]['bytes']
        unpack_char = DATA_TYPES[self._data_type]['unpack_char']

        # Preallocate a NumPy array for each channel
        for ch in range(channels):
            self.channels[ch].values = np.zeros(total_frames, dtype=np.float32)
        specific_channels = range(channels) if self._channels_to_read is None else self._channels_to_read

        # Read and unpack data in batches
        for i, frame_group in enumerate(data_order):
            for ch in range(channels):
                batch_size = len(frame_group) * pts_per_frame * data_type_bytes
                if ch in specific_channels:
                    batch_data = file_handle.read(batch_size)
                    data_format = f'<{len(frame_group) * pts_per_frame}{unpack_char}'
                    unpacked_data = struct.unpack(data_format, batch_data)
                    start_index = i * pts_per_frame * len(frame_group)
                    end_index = start_index + len(frame_group) * pts_per_frame
                    self.channels[ch].values[start_index:end_index] = unpacked_data
                else:
                    file_handle.seek(batch_size, 1)

        # Remove extra frames if needed
        if remove_last_frame:
            for ch in range(channels):
                if len(self.channels[ch].values) > 0:
                    self.channels[ch].values = self.channels[ch].values[0:pts_per_frame * frames]

        # Scale channel data
        for ch in self.channels:
            ch._apply_scale()

        # Retain only specified channels if provided
        if self._channels_to_read is not None:
            indices_to_leave = set(self._channels_to_read)
            self.channels = [ch for i, ch in enumerate(self.channels) if i in indices_to_leave]

        return True

