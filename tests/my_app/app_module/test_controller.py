import pytest
from unittest.mock import patch, MagicMock
from my_app.app_module.controller import store_token, evaluate_parameters_model_run, get_models, storage_content, eval_params_eval_dataset, eval_metrics, TOKENS
import os
from datetime import datetime
from my_app.app_module.controller import predict_audios, TOKENS
# 1. Test `store_token`
def test_store_token():
    analysis_id = "123"
    token = "Bearer abcdef123456"
    store_token(analysis_id, token)
    assert TOKENS[analysis_id] == "abcdef123456"


# 2. Test `evaluate_parameters_model_run`
@patch("my_app.model_module.prediction_pipeline.model_factory.ModelFactory.create_model", return_value=MagicMock())
@patch("my_app.model_module.prediction_pipeline.model_factory.PredictionPipeline.__init__", return_value=None)  # Mock PredictionPipeline constructor
@patch("os.path.isfile", return_value=True)  # Simulate that all files exist
@patch("os.getenv", return_value="/fake_storage_path")
def test_evaluate_parameters_model_run_valid(mock_getenv, mock_isfile, mock_pipeline_init, mock_create_model):
    # Call the function
    status, message = evaluate_parameters_model_run("any_model_name", ["audio1.wav", "audio2.wav"])
    print(f"Status: {status}, Message: {message}")

    # Assertions
    assert status is True, f"Expected True but got {status}"
    assert message == "2 passed to analysis", f"Expected '2 passed to analysis' but got '{message}'"
@patch("my_app.model_module.prediction_pipeline.model_factory.PredictionPipeline", side_effect=ValueError("Unknown model"))
def test_evaluate_parameters_model_run_invalid_model(mock_prediction_pipeline):
    status, message = evaluate_parameters_model_run("invalid_model", ["audio1.wav"])
    assert status is False
    assert "Unknown model: invalid_model" in message

@patch("my_app.model_module.prediction_pipeline.model_factory.PredictionPipeline.__init__", return_value=None)  # Mock PredictionPipeline constructor
@patch("os.path.isfile", return_value=False)  # Files do not exist in storage
@patch("os.getenv", return_value="/fake_storage_path")
def test_evaluate_parameters_model_run_missing_files(mock_getenv, mock_isfile, mock_pipeline_init):
    status, message = evaluate_parameters_model_run("valid_model", ["missing_file.wav"])
    assert status is False
    assert message == "File does not exists in storage"


# 3. Test `get_models`
@patch("my_app.model_module.prediction_pipeline.model_factory.ModelFactory.get_available_models")
def test_get_models(mock_get_available_models):
    mock_get_available_models.return_value = ["model1", "model2"]
    models = get_models()
    assert models == ["model1", "model2"]


# 4. Test `storage_content`
@patch("os.path.exists", side_effect=lambda x: x in ["/fake_storage_path/audio1.wav", "/fake_storage_path/audio2.wav"])
@patch("os.path.join", side_effect=lambda *args: "/fake_storage_path/" + args[-1])
@patch("os.getenv", return_value="/fake_storage_path")
@patch("os.listdir", return_value=["audio1.wav", "audio2.wav", "audio3.wav"])
@patch("os.path.isfile", side_effect=lambda x: x in ["/fake_storage_path/audio1.wav", "/fake_storage_path/audio2.wav"])
def test_storage_content_files_exist(mock_getenv, mock_join, mock_exists, mock_listdir, mock_isfile):
    # Call the storage_content function with the test files
    results = storage_content(["audio1.wav", "audio2.wav", "audio4.wav"])

    # Assert the expected output
    assert results == {"audio1.wav": True, "audio2.wav": True, "audio4.wav": False}

@patch("os.path.exists", side_effect=lambda x: False)
@patch("os.getenv", return_value="/fake_storage_path")
@patch("os.listdir", return_value=[])  # Mock listdir to return an empty list
def test_storage_content_files_missing(mock_getenv, mock_exists, mock_listdir):
    results = storage_content(["nonexistent.wav"])
    assert results == {"nonexistent.wav": False}

# 5. Test `eval_params_eval_dataset`
@patch("os.getenv", return_value="/fake_config_path")
def test_eval_params_eval_dataset_valid(mock_getenv):
    status, message = eval_params_eval_dataset("deep_voice", "config_mesonet.yaml")
    assert status is True
    assert "Analysis started for dataset deep_voice" in message

@patch("os.getenv", return_value="/fake_config_path")
def test_eval_params_eval_dataset_invalid_dataset(mock_getenv):
    status, message = eval_params_eval_dataset("invalid_dataset", "config_mesonet.yaml")
    assert status is False
    assert "Not correct dataset_name" in message

@patch("os.getenv", return_value="/fake_config_path")
def test_eval_params_eval_dataset_invalid_config(mock_getenv):
    status, message = eval_params_eval_dataset("deep_voice", "invalid_config.yaml")
    assert status is False
    assert "Not correct config_path" in message


# 6. Test `eval_metrics`
@patch("my_app.utils.get_scores_from_csv")
@patch("my_app.model_module.metrics.calculate_eer_from_scores", return_value=(0.05, []))
@patch("os.getenv", return_value="/fake_results_path")  # Add mock_getenv here
def test_eval_metrics_success(mock_getenv, mock_calculate_eer, mock_get_labels_and_predictions_from_csv):
    mock_get_labels_and_predictions_from_csv.return_value = ([1, 0, 1], [0, 0, 1])
    result = eval_metrics("test_dataset", "output")
    assert result["status"] == "success"
    assert result["info"] == "err calculated"
    assert result["results"] == 0.05

@patch("my_app.utils.get_scores_from_csv", side_effect=FileNotFoundError("File not found"))
@patch("os.getenv", return_value="/fake_results_path")  # Add mock_getenv here
def test_eval_metrics_file_not_found(mock_getenv, mock_get_labels_and_predictions_from_csv):
    result = eval_metrics("test_dataset", "output")
    assert result["status"] == "failure"
    assert "File not found" in result["info"]




@patch("my_app.app_module.controller.PredictionPipeline.__init__", return_value=None)
@patch("my_app.app_module.controller.predict")
@patch("my_app.app_module.controller.metrics.file_score_and_label")
@patch("my_app.app_module.controller.client_API.connector_create_predictions", return_value=(200, "Success"))
@patch("my_app.app_module.controller.client_API.connector_end_analysis", return_value=(200, "Success"))
@patch("my_app.app_module.controller.client_API.connector_abort_analysis")
@patch("my_app.app_module.controller.utils.delate_file_from_storage")
def test_predict_audios_success(
    mock_delete_file,
    mock_abort_analysis,
    mock_end_analysis,
    mock_create_predictions,
    mock_file_score_and_label,
    mock_predict,
    mock_pipeline_init,
):
    # Set up the mock data
    analysis_id = "12345"
    selected_model = "test_model"
    file_paths = ["audio1.wav", "audio2.wav"]
    TOKENS[analysis_id] = "test_token"

    # Mock predict function output
    mock_predict.return_value = (["audio1.wav"], [0, 1], [1, 0])  # segments_list and labels

    # Mock file_score_and_label function output
    mock_file_score_and_label.return_value = (0.5, 0)  # score and label

    # Mock timestamp to ensure consistency
    mock_timestamp = "2024-11-18T17:44:00.590503"
    with patch("my_app.app_module.controller.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime.fromisoformat(mock_timestamp)
        mock_datetime.now.isoformat.return_value = mock_timestamp

        # Run the function
        predict_audios(analysis_id, selected_model, file_paths)

    # Assertions to verify behavior
    assert mock_create_predictions.call_count == 2
    mock_create_predictions.assert_any_call(
        analysis_id="12345",
        payload={
            "link": "audio1.wav",
            "timestamp": mock_timestamp,
            "model": selected_model,
            "modelPredictions": [
                {"segmentNumber": 0, "label": 1},
                {"segmentNumber": 1, "label": 0},
            ],
            "score": 0.5,
            "label": 0,
        },
        token="test_token",
    )
    mock_delete_file.assert_any_call("audio1.wav")
    mock_end_analysis.assert_called_once_with(analysis_id=analysis_id, token="test_token")
    mock_abort_analysis.assert_not_called()  # Ensure no abort if successful

@patch("my_app.app_module.controller.PredictionPipeline.__init__", return_value=None)
@patch("my_app.app_module.controller.predict")
@patch("my_app.app_module.controller.client_API.connector_create_predictions", return_value=(500, "Error"))
@patch("my_app.app_module.controller.client_API.connector_abort_analysis")
@patch("my_app.app_module.controller.utils.delate_file_from_storage")
def test_predict_audios_with_error_in_predictions(
    mock_delete_file,
    mock_abort_analysis,
    mock_create_predictions,
    mock_predict,
    mock_pipeline_init,
):
    # Set up the mock data
    analysis_id = "12345"
    selected_model = "test_model"
    file_paths = ["audio1.wav"]
    TOKENS[analysis_id] = "test_token"

    # Mock predict function output
    mock_predict.return_value = (["audio1.wav"], [0, 1], ["label1", "label2"])

    # Run the function
    predict_audios(analysis_id, selected_model, file_paths)

    # Assertions to verify behavior
    mock_create_predictions.assert_called_once()
    mock_abort_analysis.assert_called_once_with(analysis_id, "test_token")
    mock_delete_file.assert_called_once_with("audio1.wav")
