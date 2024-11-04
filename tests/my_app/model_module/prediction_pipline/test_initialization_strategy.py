import unittest
from unittest.mock import patch, MagicMock
import torch
from my_app.model_module.models.meso import meso_net
import os
from my_app.model_module.prediction_pipline.initialization_strategy import MesonetInitialization, Wav2vecInitialization
from my_app.model_module.models.wav2vec.model import Model
class TestModelInitialization(unittest.TestCase):
    
    @patch("my_app.model_module.prediction_pipline.initialization_strategy.load_config")
    @patch("torch.load")
    def test_mesonet_initialization(self, mock_torch_load, mock_load_config):
        # Mock the configuration load
        mock_load_config.return_value = {
            'checkpoint': {'path': 'path/to/mesonet/checkpoint'},
            'model': {
                'parameters': {
                    'input_channels': 3,
                    'fc1_dim': 512,
                    'frontend_algorithm': 'mfcc'
                }
            }
        }

        # Mock the model's load_state_dict method
        mock_model = MagicMock()
        with patch("my_app.model_module.models.meso.meso_net.FrontendMesoInception4", return_value=mock_model):
            mesonet_init = MesonetInitialization()
            model = mesonet_init.initialize("config_files/config_mesonet.yaml")

            # Check if configuration was loaded correctly
            mock_load_config.assert_called_once_with("config_files/config_mesonet.yaml")

            # Check if model was instantiated with the correct parameters
            meso_net.FrontendMesoInception4.assert_called_once_with(
                input_channels=3, fc1_dim=512, frontend_algorithm="mfcc", device=mesonet_init.device
            )

            # Verify torch.load was called with the correct path
            mock_torch_load.assert_called_once_with("path/to/mesonet/checkpoint", map_location=mesonet_init.device)

            # Check if model is returned
            self.assertEqual(model, mock_model.to.return_value)

if __name__ == '__main__':
    unittest.main()
