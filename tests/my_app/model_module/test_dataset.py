import pytest
import numpy as np
import torch
from torch import Tensor
from unittest.mock import patch
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
    return np.random.randn(64600 * 2 + 5000)  # 2 full chunks and a remainder


@pytest.fixture
def mock_base_dir(tmp_path):
    """Temporary directory for storing mock audio files."""
    return tmp_path


@pytest.fixture
def preload_audio_files(sample_list_IDs, mock_audio_data, mock_base_dir):
    """Create mock audio files in the temporary base directory."""
    for utt_id in sample_list_IDs:
        file_path = mock_base_dir / utt_id
        np.save(file_path, mock_audio_data)  # Save as .npy files to simulate data
    return mock_base_dir


def test_dataset_custom_initialization(sample_list_IDs, mock_base_dir):
    """Test initialization and dataset preparation in Dataset_Custom."""
    with patch("os.path.exists", return_value=True), \
         patch("librosa.load", side_effect=lambda path, sr: (np.random.randn(64600 * 2 + 5000), 16000)):
        
        dataset = Dataset_Custom(list_IDs=sample_list_IDs, base_dir=str(mock_base_dir))

    # Verify the number of segments calculated based on audio data length and chunk size
    expected_segments = 3 * len(sample_list_IDs)  # 2 full chunks + 1 remainder for each file
    assert len(dataset.audio_segments) == expected_segments
    assert len(dataset) == expected_segments


def test_dataset_custom_getitem(sample_list_IDs, mock_base_dir):
    """Test the __getitem__ method in Dataset_Custom."""
    with patch("os.path.exists", return_value=True), \
         patch("librosa.load", side_effect=lambda path, sr: (np.random.randn(64600 * 2 + 5000), 16000)):

        dataset = Dataset_Custom(list_IDs=sample_list_IDs, base_dir=str(mock_base_dir))

    # Retrieve an item from the dataset
    x_inp, (utt_id, segment_num) = dataset[0]

    # Check that the output is a Tensor and has the correct shape
    assert isinstance(x_inp, Tensor)
    assert x_inp.shape[0] == 64600  # Should be the size of each chunk (cut length)

    # Check that utt_id and segment_num are correctly returned
    assert utt_id == "audio1.wav"
    assert segment_num == 0


def test_dataset_custom_length_and_segments(sample_list_IDs, mock_base_dir):
    """Test that dataset length matches the number of prepared segments."""
    with patch("os.path.exists", return_value=True), \
         patch("librosa.load", side_effect=lambda path, sr: (np.random.randn(64600 * 2 + 5000), 16000)):

        dataset = Dataset_Custom(list_IDs=sample_list_IDs, base_dir=str(mock_base_dir))

    # Each file has 2 full chunks and 1 partial chunk
    expected_length = 3 * len(sample_list_IDs)
    assert len(dataset) == expected_length
    assert len(dataset.audio_segments) == expected_length

def test_dataset_custom_missing_file(sample_list_IDs, mock_base_dir):
    """Test that a missing file raises FileExistsError and is handled gracefully."""
    # Mock os.path.exists to simulate a missing file for the first file and existing file for the second
    with patch("os.path.exists", side_effect=[False, True]), \
         patch("os.listdir", return_value=["audio2.wav"]):
        
        # Check that FileExistsError is raised when a file is missing
        with pytest.raises(FileExistsError) as exc_info:
            Dataset_Custom(list_IDs=sample_list_IDs, base_dir=str(mock_base_dir))

        # Verify that the error message contains the missing file name
        assert sample_list_IDs[0] in str(exc_info.value)


