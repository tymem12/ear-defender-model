import pytest
from unittest import mock
from my_app.app_module import client_API
from my_app.model_module import metrics
from my_app.app_module.controller import (
    store_token,
    evaluate_parameters_model_run,
    get_models,
    predict_audios,
    storage_content,
    eval_params_eval_dataset,
    eval_dataset,
    eval_metrics,
    TOKENS
)

# Mock data for testing
mock_analysis_id = "test_analysis"
mock_token = "mock_token"
mock_selected_model = "mesonet"
mock_file_paths = ["file1.wav", "file2.wav"]
mock_dataset = "test_dataset"
mock_model_conf = "config_mesonet.yaml"
mock_output_csv = "test_output"

@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to set environment variables."""
    monkeypatch.setenv("AUDIO_STORAGE", "/mock/storage")
    monkeypatch.setenv("CONFIG_FOLDER", "/mock/config")
    monkeypatch.setenv("RESULTS_CSV", "/mock/results")


@pytest.fixture
def reset_tokens():
    """Fixture to reset TOKENS dictionary after each test."""
    TOKENS.clear()


def test_store_token(reset_tokens):
    store_token(mock_analysis_id, f"Bearer {mock_token}")
    assert TOKENS[mock_analysis_id] == mock_token


def test_evaluate_parameters_model_run_valid_model(mock_env):
    with mock.patch("my_app.model_module.prediction_pipline.model_factory.PredictionPipeline") as mock_pipeline, \
         mock.patch("os.path.isfile", return_value=True):  # Mock os.path.isfile to return True for all files
        mock_pipeline.return_value = mock.Mock()  # Mock successful initialization of PredictionPipeline
        success, message = evaluate_parameters_model_run(mock_selected_model, mock_file_paths)
        print('AAAAAAAAAAAAA' + message)
        assert success is True
        assert message == f"{len(mock_file_paths)} passed to analysis"


def test_evaluate_parameters_model_run_invalid_model(mock_env):
    with mock.patch("my_app.model_module.prediction_pipline.model_factory.PredictionPipeline", side_effect=ValueError("Unknown model")):
        success, message = evaluate_parameters_model_run("invalid_model", mock_file_paths)
        assert success is False
        assert "Unknown model" in message


def test_get_models():
    with mock.patch("my_app.model_module.prediction_pipline.model_factory.ModelFactory.get_available_models", return_value=["model1", "model2"]):
        models = get_models()
        assert models == ["model1", "model2"]


@mock.patch("my_app.app_module.client_API.connector_create_predictions")
@mock.patch("my_app.utils.delate_file_from_storage")
def test_predict_audios_success(mock_delete_file, mock_connector_create_predictions, reset_tokens, mock_env):
    store_token(mock_analysis_id, f"Bearer {mock_token}")
    mock_connector_create_predictions.return_value = (200, "Success")
    
    with mock.patch("my_app.model_module.prediction_pipline.model_factory.PredictionPipeline") as mock_pipeline:
        mock_pipeline.return_value = mock.Mock()  # Mock the prediction pipeline
        predict_audios(mock_analysis_id, mock_selected_model, mock_file_paths)
    
    assert mock_connector_create_predictions.call_count == len(mock_file_paths)
    assert mock_delete_file.call_count == len(mock_file_paths)


@mock.patch("my_app.app_module.client_API.connector_create_predictions")
@mock.patch("my_app.utils.delate_file_from_storage")
def test_predict_audios_abort_on_failure(mock_delete_file, mock_connector_create_predictions, reset_tokens, mock_env):
    store_token(mock_analysis_id, f"Bearer {mock_token}")
    # Simulate a failure response for the first file
    mock_connector_create_predictions.side_effect = [(500, "Failure"), (200, "Success")]
    
    with mock.patch("my_app.model_module.prediction_pipline.model_factory.PredictionPipeline") as mock_pipeline:
        mock_pipeline.return_value = mock.Mock()  # Mock the prediction pipeline
        predict_audios(mock_analysis_id, mock_selected_model, mock_file_paths)
    
    # First call should trigger abort
    assert mock_connector_create_predictions.call_count == 1
    mock_delete_file.assert_called_once()


def test_storage_content_existing_files(mock_env):
    with mock.patch("os.path.exists", return_value=True):
        results = storage_content(mock_file_paths)
        assert all(results[file] is True for file in mock_file_paths)


def test_storage_content_missing_files(mock_env):
    with mock.patch("os.path.exists", side_effect=[True, False]):
        results = storage_content(mock_file_paths)
        assert results[mock_file_paths[0]] is True
        assert results[mock_file_paths[1]] is False


def test_eval_params_eval_dataset_valid(mock_env):
    success, message = eval_params_eval_dataset(mock_dataset, mock_model_conf)
    assert success is True
    assert "Analysis started for dataset" in message


def test_eval_params_eval_dataset_invalid_dataset(mock_env):
    success, message = eval_params_eval_dataset("invalid_dataset", mock_model_conf)
    assert success is False
    assert "Not correct dataset_name" in message


def test_eval_params_eval_dataset_invalid_config(mock_env):
    success, message = eval_params_eval_dataset(mock_dataset, "invalid_config.yaml")
    assert success is False
    assert "Not correct config_path" in message


@mock.patch("my_app.utils.get_files_to_predict", side_effect=[(["fake_file1.wav"], "/fake/path"), (["real_file1.wav"], "/real/path")])
@mock.patch("my_app.utils.save_results_to_csv")
@mock.patch("my_app.model_module.prediction_pipline.model_factory.PredictionPipeline")
def test_eval_dataset_success(mock_get_files, mock_save_results, mock_prediction_pipeline, mock_env):
    result = eval_dataset(mock_dataset, mock_model_conf, mock_output_csv)
    assert result["status"] == "success"
    assert "files saved in" in result["info"]
    assert mock_save_results.call_count == 2


@mock.patch("my_app.utils.get_labels_and_predictions_from_csv", return_value=([0, 1, 1, 0], [0, 1, 0, 1]))
def test_eval_metrics_success(mock_get_labels_and_predictions, mock_env):
    result = eval_metrics(mock_dataset, mock_output_csv)
    assert result["status"] == "success"
    assert "err calculated" in result["info"]
    assert "results" in result


@mock.patch("my_app.utils.get_labels_and_predictions_from_csv", side_effect=FileNotFoundError("File not found"))
def test_eval_metrics_failure(mock_get_labels_and_predictions, mock_env):
    result = eval_metrics(mock_dataset, mock_output_csv)
    assert result["status"] == "failure"
    assert "File not found" in result["info"]

