import unittest
from unittest.mock import Mock, patch
from my_app.model_module.prediction_pipline.base_models import Model, MesonetModel, Wav2wec  # Replace 'your_module' with the actual module name

class TestModel(unittest.TestCase):

    def setUp(self):
        # Mock initialization strategy
        self.mock_strategy = Mock()
        self.mock_strategy.initialize.return_value = lambda x: f"initialized with {x}"
        self.mock_strategy.config = {'model': {'threshold': 0.5}}

    def test_mesonet_model_default_config(self):
        model = MesonetModel(initialization_strategy=self.mock_strategy)
        self.assertEqual(model.config_path, 'config_files/config_mesonet.yaml')

    def test_wav2wec_default_config(self):
        model = Wav2wec(initialization_strategy=self.mock_strategy)
        self.assertEqual(model.config_path, 'config_files/config_wav2vec.yaml')

    def test_predict_method_mesonet_model(self):
        model = MesonetModel(initialization_strategy=self.mock_strategy)
        model.initialized_model = Mock(return_value="prediction result")
        result = model.predict("input data")
        model.initialized_model.assert_called_once_with("input data")
        self.assertEqual(result, "prediction result")

    def test_predict_method_wav2wec(self):
        model = Wav2wec(initialization_strategy=self.mock_strategy)
        model.initialized_model = Mock(return_value="prediction result")
        result = model.predict("input data")
        model.initialized_model.assert_called_once_with("input data")
        self.assertEqual(result, "prediction result")

    def test_wav2wec_get_threshold_value(self):
        model = Wav2wec(initialization_strategy=self.mock_strategy)
        threshold = model.get_threshold_value()
        self.assertEqual(threshold, 0.5)

    def test_initialize_without_strategy_raises_error(self):
        with self.assertRaises(ValueError):
            MesonetModel().initialize_model()

    @patch('my_app.model_module.prediction_pipline.base_models.Wav2wec.get_default_config', return_value='config_files/config_wav2vec.yaml')
    def test_default_config_path_inheritance(self, mock_get_default_config):
        model = Wav2wec(initialization_strategy=self.mock_strategy)
        self.assertEqual(model.config_path, 'config_files/config_wav2vec.yaml')
        mock_get_default_config.assert_called_once()


if __name__ == "__main__":
    unittest.main()
