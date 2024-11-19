import pytest
from unittest.mock import Mock, patch
from my_app.model_module.prediction_pipeline.base_models import Model, MesonetModel, Wav2wec

# Test fixtures
@pytest.fixture
def mock_initialization_strategy():
    """Creates a mock initialization strategy."""
    strategy = Mock()
    strategy.initialize = Mock(return_value=lambda x: f"Processed {x}")
    strategy.config = {'model': {'threshold': 0.5}}
    return strategy


@pytest.fixture
def mesonet_model(mock_initialization_strategy):
    """Creates a MesonetModel with the mock initialization strategy."""
    return MesonetModel(initialization_strategy=mock_initialization_strategy)


@pytest.fixture
def wav2wec_model(mock_initialization_strategy):
    """Creates a Wav2wec model with the mock initialization strategy."""
    return Wav2wec(initialization_strategy=mock_initialization_strategy)


# Test Model Base Class
def test_model_initialization_without_strategy():
    """Tests that initializing a Model without a strategy raises an error."""
    with pytest.raises(TypeError):
        model = Model()  # Model is abstract and cannot be instantiated


# Test MesonetModel
def test_mesonet_default_config(mesonet_model):
    """Tests that the MesonetModel returns the correct default config path."""
    assert mesonet_model.get_default_config() == 'config_files/config_mesonet.yaml'


def test_mesonet_initialization_with_strategy(mesonet_model, mock_initialization_strategy):
    """Tests that the MesonetModel initializes with a strategy."""
    assert mesonet_model.initialized_model("input_data") == "Processed input_data"
    mock_initialization_strategy.initialize.assert_called_once_with(mesonet_model.config_path)


def test_mesonet_prediction(mesonet_model):
    """Tests the predict method of MesonetModel."""
    result = mesonet_model.predict("test_input")
    assert result == "Processed test_input"


# Test Wav2wec Model
def test_wav2wec_default_config(wav2wec_model):
    """Tests that the Wav2wec model returns the correct default config path."""
    assert wav2wec_model.get_default_config() == 'config_files/config_wav2vec.yaml'


def test_wav2wec_initialization_with_strategy(wav2wec_model, mock_initialization_strategy):
    """Tests that the Wav2wec model initializes with a strategy."""
    assert wav2wec_model.initialized_model("input_data") == "Processed input_data"
    mock_initialization_strategy.initialize.assert_called_once_with(wav2wec_model.config_path)


def test_wav2wec_prediction(wav2wec_model):
    """Tests the predict method of Wav2wec."""
    result = wav2wec_model.predict("test_input")
    assert result == "Processed test_input"


def test_wav2wec_get_threshold_value(wav2wec_model):
    """Tests the get_threshold_value method of Wav2wec."""
    assert wav2wec_model.get_threshold_value() == 0.5


# Test Initialization without Strategy
def test_initialization_without_strategy():
    """Tests initializing MesonetModel or Wav2wec without an initialization strategy."""
    with pytest.raises(ValueError, match="Initialization strategy is not provided."):
        MesonetModel(config_path="dummy_config.yaml").initialize_model()
    with pytest.raises(ValueError, match="Initialization strategy is not provided."):
        Wav2wec(config_path="dummy_config.yaml").initialize_model()


# Parameterized Tests for Configurations
@pytest.mark.parametrize("model_class, expected_config", [
    (MesonetModel, 'config_files/config_mesonet.yaml'),
    (Wav2wec, 'config_files/config_wav2vec.yaml')
])
def test_get_default_config(model_class, expected_config, mock_initialization_strategy):
    """Parameterized test to check get_default_config for each model."""
    model_instance = model_class(initialization_strategy=mock_initialization_strategy)
    assert model_instance.get_default_config() == expected_config
