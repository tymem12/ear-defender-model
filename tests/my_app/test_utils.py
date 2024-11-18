import pytest
from unittest import mock
from unittest.mock import mock_open, patch, MagicMock, call
from io import StringIO
import csv

from my_app.utils import (
    load_config,
    get_files_to_predict,
    save_results_to_csv,
    get_labels_and_predictions_from_csv,
    get_scores_from_csv,
    delate_file_from_storage
)
def test_load_config():
    yaml_data = "key: value"
    with patch("builtins.open", mock_open(read_data=yaml_data)):
        with patch("yaml.safe_load") as mock_safe_load:
            mock_safe_load.return_value = {"key": "value"}
            config = load_config("config.yaml")
            assert config == {"key": "value"}
            mock_safe_load.assert_called_once()


@patch("os.path.isfile", return_value=True)  # Mock isfile to always return True
@patch("os.getenv", return_value="/mock_audio_datasets")  # Mock getenv
@patch("builtins.open", new_callable=mock_open, read_data="row1,row2,row3\nfile1,unused,spoof\nfile2,unused,spoof\n")
def test_get_files_to_predict(mock_open, mock_getenv, mock_isfile):

    file_list, folder = get_files_to_predict("release_in_the_wild", "fake")
    assert file_list == ["file1", "file2"]
    assert folder == "/mock_audio_datasets/release_in_the_wild"

    # Ensure the mocks were called as expected
    mock_isfile.assert_called_once_with("/mock_audio_datasets/release_in_the_wild/meta.csv")
    mock_open.assert_called_once_with("/mock_audio_datasets/release_in_the_wild/meta.csv", mode='r')
    mock_getenv.assert_called_once_with("AUDIO_DATASETS")


@patch("os.path.isfile", return_value=True)  # Mock isfile to always return True
@patch("os.getenv", return_value="/mock_audio_datasets")  # Mock getenv
@patch("builtins.open", new_callable=mock_open, read_data="row1,row2\nfile6,unused\nfile7,unused\nfile8,unused\n")
def test_get_files_to_predict_custom_dataset_train(mock_open, mock_getenv, mock_isfile):
    file_list, folder = get_files_to_predict("custom_dataset", "train")
    
    # Check that the correct list of files and folder path are returned
    assert file_list == ["file6", "file7", "file8"]
    assert folder == "/mock_audio_datasets/custom_dataset/train"
    
    # Verify the function calls for opening the correct file and fetching the environment variable
    mock_open.assert_called_once_with("/mock_audio_datasets/custom_dataset/custom_dataset_data_train.csv", mode='r')
    mock_getenv.assert_called_once_with("AUDIO_DATASETS")
    
# Test get_files_to_predict function
@patch("os.path.isfile", return_value=False)
@patch("builtins.open", new_callable=mock_open)
def test_save_results_to_csv(mock_open, mock_isfile):
    file_names = ["file1", "file2"]
    fragments = ["fragment1", "fragment2"]
    scores = [0.9, 0.8]
    labels = [1, 0]

    with patch('csv.writer') as mock_csv_writer:
        # Mock the writer instance
        mock_writer_instance = mock_csv_writer.return_value

        # Call the function under test
        save_results_to_csv(file_names, fragments, scores, labels, "output.csv")

        # Verify that the CSV writer was created with the file handle
        mock_csv_writer.assert_called_once_with(mock_open.return_value)
        
        # Verify that writer.writerow was called with the expected header and rows
        expected_calls = [
            call(['name', 'fragment', 'score', 'label']),
            call(['file1', 'fragment1', 0.9, 1]),
            call(['file2', 'fragment2', 0.8, 0])
        ]
        mock_writer_instance.writerow.assert_has_calls(expected_calls, any_order=False)
# Test get_labels_and_predictions_from_csv function
@patch("os.path.isfile", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="row1,row2,row3,row4\nid1,data1,data2,0\nid2,data2,data3,1\n")
def test_get_labels_and_predictions_from_csv(mock_open, mock_isfile):
    spoof_links = ["spoof.csv"]
    real_links = ["real.csv"]
    predictions, labels = get_labels_and_predictions_from_csv(spoof_links, real_links)
    
    assert predictions == [0, 1,0,1]
    assert labels == [0, 0,1,1]
    assert mock_isfile.call_count == 2
    assert mock_open.call_count == 2

# Test get_scores_from_csv function
@patch("os.path.isfile", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="row1,row2,ro3,row4\nid1,data1,0.9,label\nid2,data2,0.8,label\n")
def test_get_scores_from_csv(mock_open, mock_isfile):
    spoof_links = ["spoof.csv"]
    real_links = ["real.csv"]
    scores_spoof, scores_real = get_scores_from_csv(spoof_links, real_links)
    
    assert scores_spoof == [0.9, 0.8]
    assert scores_real == [0.9, 0.8]
    assert mock_isfile.call_count == 2
    assert mock_open.call_count == 2

# Test delate_file_from_storage function
@patch("os.getenv", return_value="/mock_audio_storage")
@patch("os.remove")
def test_delate_file_from_storage(mock_remove, mock_getenv):
    delate_file_from_storage("file_to_delete.csv")
    
    # Verify that os.getenv was called to retrieve the environment variable
    mock_getenv.assert_called_once_with("AUDIO_STORAGE")
    
    # Verify that os.remove was called with the correct file path
    mock_remove.assert_called_once_with("/mock_audio_storage/file_to_delete.csv")