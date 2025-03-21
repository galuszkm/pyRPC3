import copy
import numpy as np
import matplotlib.pyplot as plt

class Channel:
    """
    A class representing a channel with time-series data.

    Attributes:
        number (int): The channel number.
        name (str): The name of the channel.
        units (str): The units of measurement for the channel data.
        dt (float): The time step between data points.
        values (np.ndarray): The array of data values.
    """

    def __init__(self, number: int , name: str = '', units: str = '', dt: float = None, scale: float = 1) -> None:
        """
        Initialize a Channel instance.

        Args:
            number (int): The channel number.
            name (str, optional): The name of the channel. Defaults to an empty string.
            units (str, optional): The measurement units of the channel data. Defaults to an empty string.
            dt (float, optional): The time step between data points. Defaults to None.
            scale (float, optional): The factor to be used to scale channel values. Defaults to 1.
        """
        self.number = number
        self.name = name
        self.units = units
        self.dt = dt
        self.values: np.ndarray = np.array([])
        self._scale = scale

    @property
    def name(self) -> str:
        """
        Get the channel name.

        Returns:
            str: The name of the channel.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Set the channel name.

        Args:
            value (str): The new name for the channel.

        Raises:
            ValueError: If the provided value is not a string.
        """
        if not isinstance(value, str):
            raise ValueError("Channel name must be a string.")
        self._name = value

    @property
    def number(self) -> int:
        """
        Get the channel number.

        Returns:
            int: The channel number.
        """
        return self._number

    @number.setter
    def number(self, value: int) -> None:
        """
        Set the channel number.

        Args:
            value (int): The new channel number.

        Raises:
            ValueError: If the provided value is not an integer.
        """
        if value is not None and not isinstance(value, int):
            raise ValueError("Channel number must be an integer.")
        self._number = value

    def get_max(self) -> float:
        """
        Get the maximum value from the channel's data.

        Returns:
            float: The maximum value.

        Raises:
            ValueError: If no data is available to determine the maximum value.
        """
        if self.values.size == 0:
            raise ValueError("No data available to determine maximum value.")
        return self.values.max()

    def get_min(self) -> float:
        """
        Get the minimum value from the channel's data.

        Returns:
            float: The minimum value.

        Raises:
            ValueError: If no data is available to determine the minimum value.
        """
        if self.values.size == 0:
            raise ValueError("No data available to determine minimum value.")
        return self.values.min()

    def _apply_scale(self) -> None:
        """
        Scale the channel's data values by a internal factor set during channel initialization.

        Raises:
            ValueError: If the scale factor is not numeric (int or float).
        """
        if not isinstance(self._scale, (int, float)):
            raise ValueError("Scale factor must be numeric (int or float).")
        self.values *= self._scale

    def copy(self) -> 'Channel':
        """
        Create a deep copy of the channel instance.

        Returns:
            Channel: A deep copy of the current instance.
        """
        return copy.deepcopy(self)

    def plot(self, linewidth: float = 1) -> None:
        """
        Plot the channel's data against time.

        Args:
            linewidth (float, optional): The width of the plot line. Defaults to 1.

        Raises:
            ValueError: If dt is not set or is not positive, or if there is no data to plot.
        """
        if self.dt is None or self.dt <= 0:
            raise ValueError("A valid dt (time step) must be set for plotting.")
        if self.values.size == 0:
            raise ValueError("No data to plot.")
        time = np.arange(0, len(self.values) * self.dt, self.dt)
        plt.plot(time, self.values, linewidth=linewidth)
        plt.grid(True)
        plt.xlim(time.min(), time.max())
        plt.ylim(self.get_min(), self.get_max())
        plt.title(f'Channel {self.number}: {self.name}')
        plt.xlabel('Time [s]')
        plt.ylabel(self.units)
        plt.show()

    def __repr__(self) -> str:
        """
        Return a string representation of the Channel instance.

        Returns:
            str: A string containing channel details.
        """
        return (f"Channel(number={self.number}, name='{self.name}', units='{self.units}', "
                f"dt={self.dt}, num_values={self.values.size})")
