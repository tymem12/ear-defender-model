import pytest
import numpy as np
import torch
from torch import Tensor
from unittest.mock import patch, MagicMock
from my_app.model_module.dataset import Dataset_Custom, pad
import logging

# Fixtures for the tests
@pytest.fixture
def sample_list_IDs():
    """Sample list of audio file identifiers."""
    return ["audio1.wav", "audio2.wav"]

@pytest.fixture
def mock_audio_data():
    """Fixture that provides mock audio data with a known length."""
    return np.random.randn(64600 * 2 + 5000), 16000  # 2 full chunks and a remainder

@pytest.fixture
def mock_base_dir(tmp_path):
    """Temporary directory for storing mock audio files."""
    return tmp_path

@pytest.fixture
def mock_librosa_load(mock_audio_data):
    """Mock librosa.load to return predefined mock audio data."""
    with patch("librosa.load", return_value=mock_audio_data) as mock_load:
        yield mock_load

# Tests
def test_dataset_custom_initialization(sample_list_IDs, mock_base_dir, mock_librosa_load):
    """Test initialization and dataset preparation in Dataset_Custom."""
    with patch("os.path.exists", return_value=True):
        dataset = Dataset_Custom(list_IDs=sample_list_IDs, base_dir=str(mock_base_dir))

    # Ensure librosa.load is called for each file
    assert mock_librosa_load.call_count == len(sample_list_IDs)

    # Verify the number of segments calculated based on audio data length and chunk size
    expected_segments = 3 * len(sample_list_IDs)  # 2 full chunks + 1 remainder for each file
    assert len(dataset.audio_segments) == expected_segments
    assert len(dataset) == expected_segments


def test_dataset_custom_getitem(sample_list_IDs, mock_base_dir, mock_librosa_load):
    """Test the __getitem__ method in Dataset_Custom."""
    with patch("os.path.exists", return_value=True):
        dataset = Dataset_Custom(list_IDs=sample_list_IDs, base_dir=str(mock_base_dir))

    # Retrieve an item from the dataset
    x_inp, (utt_id, segment_num) = dataset[0]

    # Check that the output is a Tensor and has the correct shape
    assert isinstance(x_inp, Tensor)
    assert x_inp.shape[0] == 64600  # Should be the size of each chunk (cut length)

    # Check that utt_id and segment_num are correctly returned
    assert utt_id == "audio1.wav"
    assert segment_num == 0


def test_dataset_custom_padding():
    """Test the pad function for handling shorter segments."""
    x_short = np.random.randn(32000)  # Less than the cut length
    padded_x = pad(x_short, max_len=64600)

    # Check that the padded segment has the correct length
    assert len(padded_x) == 64600

    # Check that padding is correctly applied
    assert np.array_equal(padded_x[:32000], x_short)  # Original data should be at the start
    assert len(padded_x[32000:]) > 0  # Ensure padding is applied


def test_dataset_custom_missing_file(sample_list_IDs, mock_base_dir, mock_librosa_load, caplog):
    """Test that a missing file is logged and handled gracefully."""
    with patch("os.path.exists", side_effect=[False, True]), patch("os.listdir", return_value=["audio2.wav"]):
        with caplog.at_level(logging.INFO):
            dataset = Dataset_Custom(list_IDs=sample_list_IDs, base_dir=str(mock_base_dir))
    
    # Check log for missing file message
    assert "File" in caplog.text
    assert "doesn't exist" in caplog.text
    assert len(dataset) == 3  # Only the existing file (audio2.wav) should have segments


def test_dataset_custom_length_and_segments(sample_list_IDs, mock_base_dir, mock_librosa_load):
    """Test that dataset length matches the number of prepared segments."""
    with patch("os.path.exists", return_value=True):
        dataset = Dataset_Custom(list_IDs=sample_list_IDs, base_dir=str(mock_base_dir))
    
    # Each file has 2 full chunks and 1 partial chunk
    expected_length = 3 * len(sample_list_IDs)
    assert len(dataset) == expected_length
    assert len(dataset.audio_segments) == expected_length
