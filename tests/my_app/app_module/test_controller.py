import unittest
from unittest.mock import patch, MagicMock
from uuid import UUID
from datetime import datetime
from my_app.model_module.evaluate_audios import predict
from my_app.app_module import client_API
from my_app import utils
from my_app.model_module import metrics
from my_app.app_module.controller import predict_audios, eval_dataset, eval_metrics  # assuming the file is named my_module.py

class TestPredictAudios(unittest.TestCase):
    
    @patch('my_app.app_module.client_API.connector_create_predictions')
    @patch('my_app.app_module.client_API.connector_update_analysis')
    @patch('my_app.model_module.prediction_pipline.model_factory.PredictionPipeline')
    @patch('my_app.model_module.evaluate_audios.predict')
    def test_predict_audios_success(self, mock_predict, mock_pipeline, mock_create_predictions, mock_update_analysis):
        analysis_id = UUID("12345678-1234-5678-1234-567812345678")
        selected_model = "mesonet"
        file_paths = ["file1.wav", "file2.wav"]

        mock_pipeline.return_value = MagicMock()
        mock_predict.return_value = (None, [0, 1, 2], [1, 0, 1])
        mock_create_predictions.return_value = {"status": "created"}
        mock_update_analysis.return_value = {"status": "updated"}
        
        result = predict_audios(analysis_id, selected_model, file_paths)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["files_count"], len(file_paths))
        self.assertEqual(result["analysis"], analysis_id)
        self.assertEqual(result["model"], selected_model)
        
        for prediction in result["predictions"]:
            self.assertEqual(prediction["model"], selected_model)
            self.assertIn("link", prediction)
            self.assertIn("timestamp", prediction)
            self.assertIsInstance(prediction["modelPredictions"], list)

    @patch('my_app.app_module.client_API.connector_create_predictions')
    @patch('my_app.app_module.client_API.connector_update_analysis')
    @patch('my_app.model_module.prediction_pipline.model_factory.PredictionPipeline')
    @patch('my_app.model_module.evaluate_audios.predict')
    def test_predict_audios_failure_model_not_found(self, mock_predict, mock_pipeline, mock_create_predictions, mock_update_analysis):
        analysis_id = UUID("12345678-1234-5678-1234-567812345678")
        selected_model = "unknown_model"
        file_paths = ["file1.wav"]

        mock_pipeline.side_effect = ValueError("Unknown model: unknown_model")
        
        result = predict_audios(analysis_id, selected_model, file_paths)
        
        self.assertEqual(result["status"], "failure")
        self.assertIn("Unknown model", result["info"])

# class TestEvalDataset(unittest.TestCase):

#     @patch('my_app.utils.load_config', return_value={'model': {'name': 'wav2vec'}})
#     @patch('my_app.utils.get_files_to_predict')
#     @patch('my_app.utils.save_results_to_csv')
#     @patch('my_app.model_module.prediction_pipline.model_factory.PredictionPipeline')
#     @patch('my_app.model_module.evaluate_audios.predict')
#     def test_eval_dataset_success(self, mock_predict, mock_pipeline, mock_save_csv, mock_get_files, mock_load_config):
#         dataset = "deep_voice"
#         model_conf = "model_config.yaml"
#         output_csv = "output_file"
        
#         mock_pipeline.return_value = MagicMock()
#         mock_get_files.side_effect = lambda dataset, status: (["file1.wav"], "folder_path")
#         mock_predict.return_value = (None, [0, 1, 2], ([1, 0, 1], ["segment1", "segment2"]))
        
#         result = eval_dataset(dataset, model_conf, output_csv)
        
#         self.assertEqual(result["status"], "success")
#         self.assertIn("files saved", result["info"])
        
#         mock_save_csv.assert_called()

    # @patch('my_app.utils.load_config', return_value={'model': {'name': 'wav2vec'}})
    # @patch('my_app.model_module.prediction_pipline.model_factory.PredictionPipeline')
    # def test_eval_dataset_invalid_dataset(self, mock_pipeline, mock_load_config):
    #     dataset = "invalid_dataset"
    #     model_conf = "model_config.yaml"
    #     output_csv = "output_file"
        
    #     with self.assertRaises(ValueError):
    #         eval_dataset(dataset, model_conf, output_csv)

class TestEvalMetrics(unittest.TestCase):

    @patch('my_app.model_module.metrics.calculate_eer_from_labels_csv')
    def test_eval_metrics_success(self, mock_calculate_eer):
        dataset = "deep_voice"
        output_csv = "output_file"
        
        mock_calculate_eer.return_value = {"EER": 0.05}
        
        result = eval_metrics(dataset, output_csv)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("err calculated", result["info"])
        self.assertEqual(result["results"], {"EER": 0.05})

if __name__ == '__main__':
    unittest.main()
