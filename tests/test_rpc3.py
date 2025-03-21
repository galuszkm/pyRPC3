import numpy as np
from src import RPC3, Channel

# -------------------------
# Dummy subclass for RPC3 testing
# -------------------------
class DummyRPC3(RPC3):
    """
    Dummy subclass of RPC3 that bypasses file reading by manually
    initializing with dummy channels.
    """
    def _read_file(self):
        """
        Override _read_file to bypass file reading and create dummy channels.
        """
        ch1 = Channel(1, "Ch1", "V", dt=0.1)
        ch1.values = np.array([1, 2, 3], dtype=np.float32)
        ch2 = Channel(2, "Ch2", "V", dt=0.1)
        ch2.values = np.array([4, 5, 6], dtype=np.float32)
        self.channels = [ch1, ch2]
        self.dt = 0.1
        return True

# -------------------------
# Pytest Unit Tests for RPC3
# -------------------------

def test_rpc3_nonexistent_file(tmp_path):
    """
    Test that initializing RPC3 with a nonexistent file logs an error.
    """
    fake_file = tmp_path / "nonexistent.rpc3"
    rpc = RPC3(str(fake_file), debug=True)
    errors = rpc.get_errors()
    assert any("File not found:" in err for err in errors)


def test_rpc3_info(capsys):
    """
    Test that the info() method prints channel information.
    """
    rpc = DummyRPC3("dummy.rpc3", debug=False)
    rpc.info()
    captured = capsys.readouterr().out
    # Check that the printed output contains the dummy channel names.
    assert "Ch1" in captured
    assert "Ch2" in captured


def test_rpc3_save(tmp_path):
    """
    Test that the save() method creates a non-empty file using self.dt and self.channels.
    """
    rpc = DummyRPC3("dummy.rpc3", debug=False)
    output_file = tmp_path / "output.rpc3"
    rpc.save(str(output_file))
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_rpc3_save_exclude(tmp_path):
    """
    Test that the save() method correctly excludes channels specified in exclude_channels.
    """
    rpc = DummyRPC3("dummy.rpc3", debug=False)
    
    # Save file with all channels.
    file_all = tmp_path / "all_channels.rpc3"
    rpc.save(str(file_all))
    size_all = file_all.stat().st_size

    # Save file excluding channel number 2.
    file_exclude = tmp_path / "exclude_channel.rpc3"
    rpc.save(str(file_exclude), exclude_channels=[2])
    size_exclude = file_exclude.stat().st_size

    # The file with one channel excluded should be smaller than the file with all channels.
    assert size_exclude < size_all


def test_get_errors_initially_empty():
    """
    Test that get_errors() returns an empty list when no errors have occurred.
    """
    rpc = DummyRPC3("dummy.rpc3", debug=False)
    errors = rpc.get_errors()
    assert isinstance(errors, list)
    # Since the dummy _read_file() does not generate errors, the errors list should be empty.
    assert len(errors) == 0
