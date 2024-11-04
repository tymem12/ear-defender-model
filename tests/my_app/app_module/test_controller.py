import unittest
from unittest.mock import patch, MagicMock
from uuid import UUID
from datetime import datetime
import os

# Import the functions to be tested
from my_app.app_module.controller import eval_metrics, eval_params_eval_dataset, evaluate_parameters_model_run, predict_audios, storage_content, store_token
from my_app.endpoints_api import eval_dataset
from my_app.model_module.evaluate_audios import predict
from my_app.model_module.prediction_pipline.model_factory import PredictionPipeline, ModelFactory
from my_app.app_module import client_API
from my_app import utils
from my_app.model_module import metrics

import unittest
from unittest.mock import patch, MagicMock
from uuid import UUID
from datetime import datetime
import os

from my_app.model_module.evaluate_audios import predict
from my_app.model_module.prediction_pipline.model_factory import PredictionPipeline, ModelFactory
from my_app.app_module import client_API
from my_app import utils
from my_app.model_module import metrics

class TestEvaluationFunctions(unittest.TestCase):

    @patch("os.getenv")
    @patch("os.path.isfile")
    def test_evaluate_parameters_model_run_success(self, mock_isfile, mock_getenv):
        mock_getenv.return_value = "/mock_storage_path"
        mock_isfile.return_value = True
        result, message = evaluate_parameters_model_run("mock_model", ["file1.wav", "file2.wav"])
        self.assertTrue(result)
        self.assertIn("passed to analysis", message)

    @patch("os.getenv")
    @patch("os.path.isfile")
    def test_evaluate_parameters_model_run_failure_file_not_exist(self, mock_isfile, mock_getenv):
        mock_getenv.return_value = "/mock_storage_path"
        mock_isfile.side_effect = [True, False]  # First file exists, second does not
        result, message = evaluate_parameters_model_run("mock_model", ["file1.wav", "file2.wav"])
        self.assertFalse(result)
        self.assertIn("File does not exists in storage", message)

    @patch.object(PredictionPipeline, '__init__', return_value=None)
    @patch("my_app.app_module.client_API.connector_create_predictions")
    @patch("my_app.app_module.client_API.connector_update_analysis")
    def test_predict_audios_failure_invalid_model(self, mock_connector_update, mock_connector_create, mock_pipeline_init):
        analysis_id = UUID("12345678-1234-5678-1234-567812345678")
        store_token(analysis_id, "mock_token")
        
        mock_pipeline_init.side_effect = ValueError("Unknown model")  # Simulate invalid model
        with self.assertRaises(ValueError):
            predict_audios(analysis_id, "invalid_model", ["file_path.wav"])

    @patch("os.getenv")
    @patch("os.path.exists")
    def test_storage_content_file_exists(self, mock_exists, mock_getenv):
        mock_getenv.return_value = "/mock_storage_path"
        mock_exists.return_value = True
        result = storage_content(["file1.wav"])
        self.assertEqual(result, {"file1.wav": True})

    @patch("os.getenv")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_storage_content_file_does_not_exist(self, mock_listdir, mock_exists, mock_getenv):
        mock_getenv.return_value = "/mock_storage_path"
        mock_exists.return_value = False
        mock_listdir.return_value = ["file1.wav", "file2.wav"]
        
        result = storage_content(["file3.wav"])
        self.assertEqual(result, {"file3.wav": False})

    def test_eval_params_eval_dataset_success(self):
        result, message = eval_params_eval_dataset("deep_voice", "config_mesonet.yaml", "output.csv")
        self.assertTrue(result)
        self.assertIn("Analysis started", message)

    def test_eval_params_eval_dataset_failure_invalid_dataset(self):
        result, message = eval_params_eval_dataset("invalid_dataset", "config_mesonet.yaml", "output.csv")
        self.assertFalse(result)
        self.assertIn("Not correct dataset_name", message)

    @patch("my_app.utils.load_config")
    @patch("my_app.utils.get_files_to_predict")
    @patch.object(PredictionPipeline, "__init__", return_value=None)
    def test_eval_dataset_success(self, mock_pipeline_init, mock_get_files, mock_load_config):
        mock_load_config.return_value = {"model": {"name": "mock_model"}}
        mock_get_files.side_effect = [(["file_fake.wav"], "/fake_folder"), (["file_real.wav"], "/real_folder")]
        
        with patch("my_app.model_module.evaluate_audios.predict", return_value=(["file"], ["seg"], [["label"]])):
            result = eval_dataset("test_dataset", "config_mesonet.yaml", "output.csv")
            self.assertEqual(result["status"], "success")

    @patch("my_app.utils.get_labels_and_predictions_from_csv")
    def test_eval_metrics_success(self, mock_get_labels):
        mock_get_labels.return_value = ([1, 0, 1], [0, 1, 1])
        with patch("my_app.model_module.metrics.calculate_eer_from_labels", return_value=0.3):
            result = eval_metrics("test_dataset", "output.csv")
            self.assertEqual(result["status"], "success")
            self.assertIn("err calculated", result["info"])

    @patch("my_app.utils.get_labels_and_predictions_from_csv", side_effect=FileNotFoundError("File not found"))
    def test_eval_metrics_file_not_found(self, mock_get_labels):
        result = eval_metrics("test_dataset", "non_existent_output")
        self.assertEqual(result["status"], "failure")
        self.assertIn("File not found", result["info"])

if __name__ == "__main__":
    unittest.main()
