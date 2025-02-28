import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for testing
import matplotlib.pyplot as plt
from src import Channel

# -------------------------
# Pytest Unit Tests for Channel
# -------------------------

def test_initialization():
    """
    Test that a Channel instance is initialized with the correct default values.
    """
    channel = Channel(number=1, name="Test", units="V", dt=0.1)
    assert channel.number == 1
    assert channel.name == "Test"
    assert channel.units == "V"
    assert channel.dt == 0.1
    np.testing.assert_array_equal(channel.values, np.array([]))


def test_name_setter_valid():
    """
    Test that setting a valid channel name works.
    """
    channel = Channel(number=2)
    channel.name = "NewName"
    assert channel.name == "NewName"


def test_name_setter_invalid():
    """
    Test that setting an invalid (non-string) channel name raises a ValueError.
    """
    channel = Channel(number=3)
    with pytest.raises(ValueError):
        channel.name = 123  # Not a string


def test_number_setter_valid():
    """
    Test that setting a valid channel number works.
    """
    channel = Channel(number=4)
    channel.number = 10
    assert channel.number == 10


def test_number_setter_invalid():
    """
    Test that setting an invalid (non-integer) channel number raises a ValueError.
    """
    channel = Channel(number=5)
    with pytest.raises(ValueError):
        channel.number = "abc"  # Not an int


def test_get_max_no_data():
    """
    Test that calling get_max on an empty data array raises a ValueError.
    """
    channel = Channel(number=6)
    channel.values = np.array([])
    with pytest.raises(ValueError):
        channel.get_max()


def test_get_min_no_data():
    """
    Test that calling get_min on an empty data array raises a ValueError.
    """
    channel = Channel(number=7)
    channel.values = np.array([])
    with pytest.raises(ValueError):
        channel.get_min()


def test_get_max_min_with_data():
    """
    Test get_max and get_min methods with valid data.
    """
    channel = Channel(number=8)
    data = np.array([1, 3, 2, 5, 4])
    channel.values = data
    assert channel.get_max() == 5
    assert channel.get_min() == 1


def test_apply_scale_valid():
    """
    Test that apply_scale properly multiplies the channel data.
    """
    channel = Channel(number=9, scale=2)
    data = np.array([1, 2, 3])
    channel.values = data.copy()
    channel._apply_scale()
    np.testing.assert_array_equal(channel.values, data * 2)


def test_apply_scale_invalid():
    """
    Test that applying a non-numeric scale factor raises a ValueError.
    """
    channel = Channel(number=10, scale="a")
    with pytest.raises(ValueError):
        channel._apply_scale()


def test_copy():
    """
    Test that copy creates a deep copy of the Channel instance.
    """
    channel = Channel(number=11, name="Test", units="V", dt=0.1)
    channel.values = np.array([1, 2, 3])
    channel_copy = channel.copy()
    # Verify that attributes are equal.
    assert channel_copy.number == channel.number
    assert channel_copy.name == channel.name
    np.testing.assert_array_equal(channel_copy.values, channel.values)
    # Modify the copy and check that the original remains unchanged.
    channel_copy.values[0] = 100
    assert channel_copy.values[0] != channel.values[0]


def test_plot_no_data():
    """
    Test that calling plot with no data raises a ValueError.
    """
    channel = Channel(number=12, name="Test", units="V", dt=0.1)
    channel.values = np.array([])
    with pytest.raises(ValueError):
        channel.plot()


def test_plot_invalid_dt():
    """
    Test that calling plot with an invalid dt raises a ValueError.
    """
    channel = Channel(number=13, name="Test", units="V", dt=0)
    channel.values = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        channel.plot()


def test_plot_success(monkeypatch):
    """
    Test that plot executes without error when provided valid data and dt.
    The plt.show function is overridden to prevent an actual plot window.
    """
    channel = Channel(number=14, name="Test", units="V", dt=0.1)
    channel.values = np.array([1, 2, 3, 4, 5])
    monkeypatch.setattr(plt, "show", lambda: None)
    # If no exception is raised, the test passes.
    channel.plot()
