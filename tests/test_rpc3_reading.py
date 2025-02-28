import pytest
from src import RPC3

@pytest.mark.parametrize("test_file", ["tests/rsp/test1.rsp"])
def test_read_rsp_file1(test_file):
    """
    Test reading an existing .rsp file. It should contain five channels
    with the following properties (rounded to 3 decimal places):

        1. FDO_54xLoc_sh   units=N,      max=241.96,	min=-220.723
        2. ACC_76zGlob	   units=m/s^2,  max=115.302,   min=88.133
        3. FFG_78zGlob	   units=N,      max=123.996,   min=93.503
        4. FAD_7yknc	   units=N,      max=155.183,   min=103.831
        5. D_23magLo	   units=mm,     max=1001.466,  min=-85.577

    Additionally, each channel should have a total duration of 8.188 seconds,
    when computed as (len(channel.values) * channel.dt) and rounded to 3 decimals.
    """

    # Expected channel info: (number, name, units, max_val, min_val)
    expected_channels = [
        (1, "FDO_54xLoc_sh", "N",      241.96,   -220.723),
        (2, "ACC_76zGlob",   "m/s^2",  115.302,	   88.133),
        (3, "FFG_78zGlob",   "N",      123.996,	   93.503),
        (4, "FAD_7yknc",     "N",      155.183,	  103.831),
        (5, "D_23magLo",     "mm",     1001.466,  -85.577),
    ]
    expected_duration = 8.188  # seconds, to 3 decimal places

    # Read the file using RPC3
    rpc = RPC3(test_file)

    # Check that we have exactly 5 channels
    assert len(rpc.channels) == 5, f"Expected 5 channels, found {len(rpc.channels)}."

    # Loop over the channels and compare properties
    for idx, channel in enumerate(rpc.channels):
        exp_num, exp_name, exp_units, exp_max, exp_min = expected_channels[idx]

        # Check channel number, name, and units
        assert channel.number == exp_num,  f"Channel {idx+1}: number mismatch."
        assert channel.name == exp_name,   f"Channel {idx+1}: name mismatch."
        assert channel.units == exp_units, f"Channel {idx+1}: units mismatch."

        # Check max and min (rounded to 3 decimal places)
        calc_max = round(channel.get_max(), 3)
        calc_min = round(channel.get_min(), 3)
        assert calc_max == pytest.approx(exp_max, 0.001), f"Channel {idx+1}: max mismatch."
        assert calc_min == pytest.approx(exp_min, 0.001), f"Channel {idx+1}: min mismatch."

        # Check total duration to 3 decimal places
        total_time = round(len(channel.values) * channel.dt, 3)
        assert total_time == pytest.approx(expected_duration, 0.001), (
            f"Channel {idx+1}: expected duration {expected_duration}, got {total_time}."
        )
