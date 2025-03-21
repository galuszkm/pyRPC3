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

You can install pyRPC3 directly from our GitLab repository using pip:

```bash
pip install git+https://cae:911/michgalu/pyRPC3.git
```

### Install for development

Clone the repository and install the required packages:

```bash
git clone https://cae:911/michgalu/pyRPC3.git
cd pyRPC3
pip install -r requirements.txt
```

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

### Testing

The repository includes a suite of tests using `pytest`. To run the tests, simply execute:

```bash
pytest
```

Tests cover:
- Reading RPC3 files and verifying channel properties.
- Writing RPC3 files and round-tripping file data.
- Data normalization and interactive plotting.

## Contributing

Contributions are welcome! To contribute:

1. **Clone the repository:**

   ```bash
   git clone https://cae:911/michgalu/pyRPC3.git
   ```

2. **Create a new branch:**

   ```bash
   git checkout -b feature/my-new-feature
   ```

3. **Make your changes and commit them:**

   ```bash
   git commit -am "Add new feature"
   ```

4. **Push to your branch:**

   ```bash
   git push origin feature/my-new-feature
   ```

5. **Open a merge request** on the GitLab repository.

