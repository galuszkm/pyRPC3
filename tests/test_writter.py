import pytest
import numpy as np
from src import RPC3, write_rpc3

@pytest.mark.parametrize("test_file", ["tests/rsp/test1.rsp"])
def test_write_rpc3_roundtrip(tmp_path, test_file):
    """
    Test that reading an existing .rsp file, writing it to a new file using write_rpc3,
    and then reading the new file back into RPC3 yields identical channel properties
    and data.

    Steps:
      1) Read the original .rsp file into an RPC3 instance.
      2) Write the channels out to a new file using write_rpc3.
      3) Read the new file into another RPC3 instance.
      4) Compare both RPC3 instances:
         - Same number of channels
         - Each channel has the same number, name, units, and dt
         - Each channel's data array is approximately equal
    """

    # Read the original file
    original_rpc = RPC3(test_file)
    # Ensure the file was read without errors
    assert not original_rpc.get_errors(), f"Errors encountered reading {test_file}: {original_rpc.get_errors()}"

    # Create a temporary output file path
    output_file = tmp_path / "roundtrip_output.rpc3"

    # Write the channels from the original RPC3 to the new file
    write_rpc3(str(output_file), original_rpc.dt, original_rpc.channels)

    # Read the newly written file
    roundtrip_rpc = RPC3(str(output_file))
    # Ensure the round-trip file was read without errors
    assert not roundtrip_rpc.get_errors(), f"Errors encountered reading {output_file}: {roundtrip_rpc.get_errors()}"

    # Compare the number of channels
    assert len(original_rpc.channels) == len(roundtrip_rpc.channels), (
        "Mismatch in the number of channels between original and roundtrip files."
    )

    # Compare each channel in order
    for idx, (orig_ch, new_ch) in enumerate(zip(original_rpc.channels, roundtrip_rpc.channels), start=1):
        # Check basic properties
        assert orig_ch.number == new_ch.number,  f"Channel {idx} number mismatch."
        assert orig_ch.name   == new_ch.name,    f"Channel {idx} name mismatch."
        assert orig_ch.units  == new_ch.units,   f"Channel {idx} units mismatch."

        # Compare dt (if dt is stored in Channel)
        # If dt is stored only in RPC3, you can compare original_rpc.dt and roundtrip_rpc.dt.
        # If dt is per channel, do an approximate comparison for floating-point consistency.
        assert abs(orig_ch.dt - new_ch.dt) < 1e-9, f"Channel {idx} dt mismatch."

        # Compare data array length
        assert len(orig_ch.values) == len(new_ch.values), (
            f"Channel {idx} length mismatch: {len(orig_ch.values)} vs {len(new_ch.values)}."
        )

        # Compare data values with a small tolerance for potential int16 normalization differences
        np.testing.assert_allclose(
            orig_ch.values,
            new_ch.values,
            rtol=1e-3,
            err_msg=f"Channel {idx} data mismatch."
        )
