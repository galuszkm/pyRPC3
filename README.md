# pyRPC3

**pyRPC3** is a Python package for reading, processing, and writing RPC3 (.rsp, .rpc, .tim) files — a binary file format used to store time-series channel data. The package provides a `Channel` class for representing individual data channels, an `RPC3` class for reading and writing RPC3 files, and utility functions for normalizing channel data.

## Features

- **Reading RPC3 Files:** Parse RPC3 files to extract header information and channel data.
- **Writing RPC3 Files:** Save processed channel data to a valid RPC3 file.
- **Data Normalization:** Normalize channel data to 16-bit integer ranges.
- **Interactive Plotting:** Visualize channel data using Matplotlib.
- **Testing:** Comprehensive tests using `pytest`.

## Installation

### Install via pip

You can install pyRPC3 directly from the GitHub repository:

```bash
pip install git+https://github.com/galuszkm/pyRPC3.git
```

### Install for development

Clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/galuszkm/pyRPC3.git
cd pyRPC3
uv run just install
```

This installs all dependencies and wires the git hooks in one step.

### Requirements

- Python 3.11+
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

**Interactive Plotting Note:**
If you need interactive plots and do not have Tkinter installed, you can use an alternative backend like PySide6. Install it with:

```bash
pip install PySide6
```

Then, set the backend in your script as follows:

```python
import matplotlib
matplotlib.use("QtAgg")  # Use an interactive Qt backend
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode
```

## Usage

### Reading an RPC3 File

To read an RPC3 file and print a summary of its channels:

```python
from pyRPC3 import RPC3

# Replace with the path to your RPC3 file
rpc = RPC3("path/to/your_file.rsp", debug=True)
rpc.info()

# Access individual channel data:
for channel in rpc.channels:
    print(f"Channel {channel.number}: {channel.name} [{channel.units}]")
```

### Writing an RPC3 File

You can write a new RPC3 file from the channels loaded in an RPC3 instance. You can also exclude certain channels by number or name:

```python
from pyRPC3 import RPC3

# Read an existing RPC3 file
rpc = RPC3("path/to/your_file.rsp", debug=True)

# Save to a new file, excluding channel number 2 and a channel named "TestChannel"
rpc.save("path/to/new_file.rsp", exclude_channels=[2, "TestChannel"])
```

### Plotting Channel Data

To plot the data of a channel using Matplotlib:

```python
import matplotlib.pyplot as plt
from pyRPC3 import RPC3

rpc = RPC3("path/to/your_file.rsp")
# Plot the first channel (ensure interactive backend is set as described above)
rpc.channels[0].plot()
plt.show()
```

## Development

### Quality Checks

```bash
uv run just check    # format + lint + type check + hooks
uv run just test     # pytest with coverage (>=70%)
uv run just format   # auto-format with Ruff
```

### Running Tests

```bash
uv run just test
```

Tests cover:
- Reading RPC3 files and verifying channel properties.
- Writing RPC3 files and round-tripping file data.
- Data normalization and interactive plotting.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and contribution guidelines.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
