import math
import numpy as np
from datetime import datetime
from .Channel import Channel

def normalize_int16(array: np.ndarray):
    """
    Normalize a NumPy array to int16 limits.

    Args:
        array (np.ndarray): Input array.

    Returns:
        tuple: Normalized array (np.ndarray) and normalization factor (float).
    """
    # Array bounds
    absmax_value = np.abs([array.max(), array.min()]).max()
    # Normalization factor
    factor = absmax_value / (2**15 - 1)
    # Normalize array
    if factor > 0:
        normalized_array = (array / factor).astype(np.int16)
    else:
        normalized_array = array.astype(np.int16)
    return normalized_array, factor

def write_rpc3(filename: str, dt: float, channels: list[Channel]):
    """
    Write RPC3 file using provided channel data.

    Args:
        filename (str): Output file name.
        dt (float): Time step.
        channels (list[Channel]): List of Channel instances.
    """
    _write_file(filename, dt, channels)

def _write_file(filename: str, dt: float, channels: list[Channel]):
    """
    Write the RPC3 file.

    Args:
        filename (str): Output file name.
        dt (float): Time step.
        channels (list[Channel]): List of Channel instances.
    """
    pts_per_frame = 1024

    # Normalize channels
    norm_channels = [normalize_int16(c.values) for c in channels]

    max_chan_len = max(len(c.values) for c in channels)
    frames = math.ceil(max_chan_len / pts_per_frame)
    pts_per_group = frames * pts_per_frame

    # Collect channel header values
    chan_head = [
        [c.name, c.units, format(norm_channels[idx][1], '8.6E')]
        for idx, c in enumerate(channels)
    ]

    header_bytes = _write_header(dt, chan_head, pts_per_frame, frames, pts_per_group)
    data_bytes = _write_data([nc[0] for nc in norm_channels], pts_per_group)

    with open(filename, 'wb') as f:
        f.write(header_bytes)
        f.write(data_bytes)

def _write_header(dt: float, chan_data: list, pts_per_frame: int, frames: int, pts_per_group: int) -> bytes:
    """
    Write the RPC3 file header.

    Args:
        dt (float): Time step.
        chan_data (list): List of channel header data.
        pts_per_frame (int): Points per frame.
        frames (int): Number of frames.
        pts_per_group (int): Points per group.

    Returns:
        bytes: Encoded header.
    """
    ctime = datetime.now()
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
    channel_keys = [
        'DESC.CHAN_',
        'UNITS.CHAN_',
        'SCALE.CHAN_',
        'LOWER_LIMIT.CHAN_',
        'UPPER_LIMIT.CHAN_',
    ]

    values = [
        'BINARY',
        str(math.ceil((len(keys) + len(channel_keys) * len(chan_data)) / 4)),
        str(len(keys) + len(channel_keys) * len(chan_data)),
        'TIME_HISTORY',
        'RESPONSE',
        format(dt, '8.6E'),
        str(len(chan_data)),
        f'{ctime.hour}:{ctime.minute}:{ctime.second} {ctime.day}-{ctime.month}-{ctime.year}',
        '1',
        'SHORT_INTEGER',
        str(pts_per_frame),
        str(pts_per_group),
        str(frames),
    ]

    # Append channel-specific header keys and values
    for idx, c_data in enumerate(chan_data):
        keys += [k + str(idx + 1) for k in channel_keys]
        values += [*c_data, '1', '-1']

    header_bytes = b''
    for k, v in zip(keys, values):
        header_bytes += k.encode().ljust(32, b'\x00') + v.encode().ljust(96, b'\x00')

    header_len = 512 * int(values[1])
    header_bytes = header_bytes.ljust(header_len, b'\x00')
    return header_bytes

def _write_data(data: list, pts_per_group: int) -> bytes:
    """
    Write the data section for the RPC3 file.

    Args:
        data (list): List of normalized channel data arrays.
        pts_per_group (int): Points per group.

    Returns:
        bytes: Encoded data.
    """
    data_bytes = b''
    for d in data:
        if len(d) < pts_per_group:
            d = np.pad(d, (0, pts_per_group - len(d)), 'constant', constant_values=(d[-1]))
        data_bytes += d.tobytes()
    return data_bytes
