import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import os
from my_app.model_module.dataset import Dataset_Custom, pad

class TestDatasetCustom(unittest.TestCase):

    @patch("os.path.exists")
    @patch("os.listdir")
    def setUp(self, mock_listdir, mock_exists):
        # Mock existence of audio files
        self.list_IDs = ["audio_1.m4a", "audio_2.m4a", "missing_audio.m4a"]
        self.base_dir = "/fake_dir"
        mock_exists.side_effect = lambda x: x != os.path.join(self.base_dir, "missing_audio.m4a")
        mock_listdir.return_value = ["audio_1.m4a", "audio_2.m4a"]

    @patch("os.path.exists", return_value=True)
    @patch("librosa.load")
    def test_prepare_dataset(self, mock_load, mock_exists):
        # Mock loading of audio files with varying lengths
        mock_load.side_effect = [
            (np.random.randn(160000), 16000),  # 10 seconds
            (np.random.randn(320000), 16000),  # 20 seconds
            (np.array([]), 16000)              # Empty (simulating missing or corrupt)
        ]

        dataset = Dataset_Custom(self.list_IDs, self.base_dir)
        self.assertEqual(len(dataset), 8, "Incorrect number of segments prepared.")
        
        # Verifying segments created for each valid file
        self.assertIn(("audio_1.m4a", 0), dataset.audio_segments)
        self.assertIn(("audio_1.m4a", 1), dataset.audio_segments)
        self.assertIn(("audio_2.m4a", 0), dataset.audio_segments)
        self.assertIn(("audio_2.m4a", 1), dataset.audio_segments)
        self.assertNotIn(("missing_audio.m4a", 0), dataset.audio_segments, "Segments should not be created for missing files.")
        
    @patch("os.path.exists", return_value=True)
    @patch("librosa.load")
    def test_getitem_with_padding(self, mock_load, mock_exists):
        # Load audio with fewer than 'cut' samples to trigger padding
        mock_load.return_value = (np.random.randn(32000), 16000)  # 2 seconds of audio
        
        dataset = Dataset_Custom(["audio_1.m4a"], self.base_dir)
        x_inp, (utt_id, segment_num) = dataset[0]
        
        self.assertEqual(x_inp.shape[0], dataset.cut, "Audio segment length should match cut size with padding.")
        self.assertEqual((utt_id, segment_num), ("audio_1.m4a", 0), "Incorrect utt_id or segment number returned.")

    @patch("os.path.exists", return_value=True)
    @patch("librosa.load")
    def test_getitem_without_padding(self, mock_load, mock_exists):
        # Load audio with exactly the cut size
        mock_load.return_value = (np.random.randn(64600), 16000)
        
        dataset = Dataset_Custom(["audio_1.m4a"], self.base_dir)
        x_inp, (utt_id, segment_num) = dataset[0]
        
        self.assertEqual(x_inp.shape[0], dataset.cut, "Audio segment length should match cut size without padding.")
        self.assertEqual((utt_id, segment_num), ("audio_1.m4a", 0), "Incorrect utt_id or segment number returned.")

    def test_padding_function(self):
        # Test padding on short array
        short_audio = np.array([0.1, 0.2, 0.3])
        padded_audio = pad(short_audio, 64600)
        
        self.assertEqual(len(padded_audio), 64600, "Padded audio length should match the max_len parameter.")
        self.assertTrue(np.allclose(padded_audio[:3], short_audio), "Padded audio should start with the original audio.")
    
    @patch("os.path.exists", return_value=True)
    @patch("librosa.load")
    def test_dataset_length_with_empty_audio(self, mock_load, mock_exists):
        # Test with an empty audio file (edge case)
        mock_load.return_value = (np.array([]), 16000)
        
        dataset = Dataset_Custom(["audio_1.m4a"], self.base_dir)
        self.assertEqual(len(dataset), 0, "Dataset length should be zero for empty audio files.")

    @patch("os.path.exists", return_value=True)
    @patch("librosa.load")
    def test_dataset_length_with_single_short_file(self, mock_load, mock_exists):
        # Test with one file, shorter than `cut`, to verify it only creates one segment
        mock_load.return_value = (np.random.randn(32000), 16000)  # Shorter than cut
        
        dataset = Dataset_Custom(["audio_1.m4a"], self.base_dir)
        self.assertEqual(len(dataset), 1, "Dataset length should be one for a single short file.")
        
    @patch("os.path.exists", return_value=True)
    @patch("librosa.load")
    def test_multiple_files_with_mixed_lengths(self, mock_load, mock_exists):
        # Test with files of various lengths
        mock_load.side_effect = [
            (np.random.randn(160000), 16000),  # 10 seconds 2,5 fragments -> 3 segments
            (np.random.randn(320000), 16000),  # 20 seconds 5 fragments -> 5 segments
            (np.random.randn(10000), 16000)    # < 1 fragments -> 1 segment
        ]
        
        dataset = Dataset_Custom(self.list_IDs[:3], self.base_dir)
        self.assertEqual(len(dataset), 9, "Dataset length should account for the total number of chunks across files.")
    

# Run the tests
if __name__ == "__main__":
    unittest.main()
