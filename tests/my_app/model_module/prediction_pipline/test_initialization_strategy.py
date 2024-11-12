import pytest
from unittest.mock import Mock, patch
import torch
from my_app.model_module.prediction_pipline.initialization_strategy import MesonetInitialization, Wav2vecInitialization


# Fixtures
@pytest.fixture
def mock_load_config():
    """Mock configuration loading."""
    return {
        'checkpoint': {'path': 'fake_checkpoint.pth'},
        'model': {
            'parameters': {
                'input_channels': 3,
                'fc1_dim': 2048,
                'frontend_algorithm': 'mfcc'
            }
        }
    }

@pytest.fixture
def mock_meso_net():
    """Mock the meso_net module and its model class."""
    with patch('my_app.model_module.models.meso.meso_net.FrontendMesoInception4') as MockMesoNet:
        mock_instance = MockMesoNet.return_value
        mock_instance.load_state_dict = Mock()
        mock_instance.to = Mock(return_value=mock_instance)  # Ensures chaining with `to()`
        yield MockMesoNet

@pytest.fixture
def mock_wav2vec_model():
    """Mock the Wav2vec Model class and its methods."""
    # Patch the Model class at the exact path where it is used in initialization_strategy.py
    with patch('my_app.model_module.prediction_pipline.initialization_strategy.Model') as MockWav2vecModel:
        # Mock the model instance
        mock_instance = MockWav2vecModel.return_value
        mock_instance.load_state_dict = Mock()  # Mock `load_state_dict` to prevent errors
        mock_instance.to = Mock(return_value=mock_instance)  # Allows chaining with `to()`
        
        # Mock the Model instantiation to return the mocked instance
        yield MockWav2vecModel

@pytest.fixture
def mock_torch_load():
    """Mock torch.load function to return a compatible state dictionary for DataParallel."""
    # Return a mock state dict structure compatible with DataParallel
    mock_state_dict = {'module.layer.weight': torch.tensor([1.0])}
    with patch('torch.load', return_value=mock_state_dict) as mock_load:
        yield mock_load

@pytest.fixture
def mock_data_parallel():
    """Mock nn.DataParallel to prevent actual parallel processing."""
    with patch('torch.nn.DataParallel') as MockDataParallel:
        mock_instance = MockDataParallel.return_value
        mock_instance.load_state_dict = Mock()  # Mock `load_state_dict`
        mock_instance.to = Mock(return_value=mock_instance)  # Allows chaining with `to()`
        yield MockDataParallel

@pytest.fixture
def mock_fairseq_checkpoint():
    """Mock fairseq's load_model_ensemble_and_task to prevent file access."""
    with patch('fairseq.checkpoint_utils.load_model_ensemble_and_task') as mock_load_model:
        # Return a tuple that simulates (model_list, cfg, task) as expected by the code
        mock_load_model.return_value = ([Mock()], Mock(), Mock())
        yield mock_load_model


# Tests for MesonetInitialization
def test_mesonet_initialization(mock_load_config, mock_meso_net, mock_torch_load):
    """Tests MesonetInitialization model setup and state dict loading."""
    with patch('my_app.model_module.prediction_pipline.initialization_strategy.load_config', return_value=mock_load_config):
        strategy = MesonetInitialization()
        model = strategy.initialize('dummy_config_path')

        # Check config loaded correctly
        assert strategy.config == mock_load_config

        # Validate meso_net.FrontendMesoInception4 was called with expected parameters
        mock_meso_net.assert_called_once_with(
            input_channels=3,
            fc1_dim=2048,
            frontend_algorithm='mfcc',
            device=strategy.device
        )

        # Validate model loading
        mock_torch_load.assert_called_once_with('fake_checkpoint.pth', map_location=strategy.device)
        model.load_state_dict.assert_called_once_with({'module.layer.weight': torch.tensor([1.0])})
        model.to.assert_called_once_with(strategy.device)


# Tests for Wav2vecInitialization
def test_wav2vec_initialization(mock_load_config, mock_wav2vec_model, mock_torch_load, mock_data_parallel, mock_fairseq_checkpoint):
    """Tests Wav2vecInitialization model setup and state dict loading."""
    with patch('my_app.model_module.prediction_pipline.initialization_strategy.load_config', return_value=mock_load_config):
        strategy = Wav2vecInitialization()
        model = strategy.initialize('dummy_config_path')

        # Check config loaded correctly
        assert strategy.config == mock_load_config

        # Validate Model initialization with the expected parameters
        mock_wav2vec_model.assert_called_once_with(args=mock_load_config['model']['parameters'], device=strategy.device)

        # Verify DataParallel wrapping and device assignment
        mock_data_parallel.assert_called_once_with(mock_wav2vec_model.return_value)
        model.to.assert_called_once_with(strategy.device)

        # Validate model loading
        mock_torch_load.assert_called_once_with('fake_checkpoint.pth', map_location=strategy.device)
        model.load_state_dict.assert_called_once_with({'module.layer.weight': torch.tensor([1.0])})


# Additional Tests for Edge Cases
def test_mesonet_initialization_missing_parameters(mock_meso_net, mock_torch_load):
    """Tests MesonetInitialization with missing model parameters in config."""
    incomplete_config = {
        'checkpoint': {'path': 'fake_checkpoint.pth'},
        'model': {'parameters': {}}
    }

    with patch('my_app.model_module.prediction_pipline.initialization_strategy.load_config', return_value=incomplete_config):
        strategy = MesonetInitialization()
        model = strategy.initialize('dummy_config_path')

        # Check that default parameters are used when missing in config
        mock_meso_net.assert_called_once_with(
            input_channels=1,  # default value
            fc1_dim=1024,  # default value
            frontend_algorithm="lfcc",  # default value
            device=strategy.device
        )


def test_wav2vec_initialization_missing_parameters(mock_wav2vec_model, mock_torch_load, mock_data_parallel, mock_fairseq_checkpoint):
    """Tests Wav2vecInitialization with missing model parameters in config."""
    incomplete_config = {
        'checkpoint': {'path': 'fake_checkpoint.pth'},
        'model': {'parameters': {}}
    }

    with patch('my_app.model_module.prediction_pipline.initialization_strategy.load_config', return_value=incomplete_config):
        strategy = Wav2vecInitialization()
        model = strategy.initialize('dummy_config_path')

        # Check that the model initializes without specific parameters and loads correctly
        mock_wav2vec_model.assert_called_once_with(args={}, device=strategy.device)
        model.load_state_dict.assert_called_once_with({'module.layer.weight': torch.tensor([1.0])})
