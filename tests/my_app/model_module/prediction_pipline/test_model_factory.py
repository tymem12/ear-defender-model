import pytest
from unittest.mock import Mock, patch
import torch
from my_app.model_module.prediction_pipline import base_models as bm
from my_app.model_module.prediction_pipline.model_factory import ModelFactory, PredictionPipeline
from my_app.model_module.prediction_pipline import postprocessing_strategy as ps
from my_app.model_module.prediction_pipline import initialization_strategy as init_strat


# Tests for ModelFactory
def test_model_factory_create_mesonet_model():
    """Test creating a Mesonet model using ModelFactory."""
    with patch.object(init_strat, 'MesonetInitialization', return_value=Mock()) as mock_init:
        with patch.object(bm, 'MesonetModel') as MockModel:
            model = ModelFactory.create_model('mesonet', 'config_path')
            mock_init.assert_called_once()
            MockModel.assert_called_once_with('config_path', mock_init.return_value)
            assert model == MockModel.return_value  # Directly compare with MockModel's return value


def test_model_factory_create_wav2vec_model():
    """Test creating a Wav2vec model using ModelFactory."""
    with patch.object(init_strat, 'Wav2vecInitialization', return_value=Mock()) as mock_init:
        with patch.object(bm, 'Wav2wec') as MockModel:
            model = ModelFactory.create_model('wav2vec', 'config_path')
            mock_init.assert_called_once()
            MockModel.assert_called_once_with('config_path', mock_init.return_value)
            assert model == MockModel.return_value  # Directly compare with MockModel's return value


def test_model_factory_unknown_model():
    """Test creating an unknown model raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model: unknown"):
        ModelFactory.create_model('unknown')


def test_model_factory_model_exists():
    """Test model_exists method for known and unknown models."""
    assert ModelFactory.model_exists('mesonet') is True
    assert ModelFactory.model_exists('wav2vec') is True
    assert ModelFactory.model_exists('unknown') is False


def test_model_factory_get_available_models():
    """Test get_available_models returns the correct list."""
    assert ModelFactory.get_available_models() == ['mesonet', 'wav2vec']


# Tests for PredictionPipeline
@pytest.fixture
def mock_mesonet_model():
    """Fixture for a mocked MesonetModel."""
    with patch.object(bm, 'MesonetModel') as MockModel:
        mock_instance = MockModel.return_value
        mock_instance.initialized_model.eval = Mock()
        mock_instance.predict = Mock(return_value=[torch.tensor([0.5])])
        yield mock_instance


@pytest.fixture
def mock_wav2vec_model():
    """Fixture for a mocked Wav2vec model with a threshold."""
    with patch.object(bm, 'Wav2wec') as MockModel:
        mock_instance = MockModel.return_value
        mock_instance.initialized_model.eval = Mock()
        mock_instance.predict = Mock(return_value=[torch.tensor([0.9])])
        mock_instance.get_threshold_value = Mock(return_value=0.5)
        yield mock_instance


def test_prediction_pipeline_mesonet(mock_mesonet_model):
    """Test PredictionPipeline with Mesonet model."""
    with patch.object(ModelFactory, 'create_model', return_value=mock_mesonet_model):
        with patch.object(ps, 'MesoPostprocessing') as MockPostprocessing:
            postprocessor_instance = MockPostprocessing.return_value
            postprocessor_instance.process = Mock(return_value='processed_output')
            
            pipeline = PredictionPipeline('mesonet', 'config_path')
            assert pipeline.model == mock_mesonet_model
            assert pipeline.postprocessing_strategy == postprocessor_instance
            
            output = pipeline.predict('input_data')
            mock_mesonet_model.predict.assert_called_once_with('input_data')
            postprocessor_instance.process.assert_called_once_with([torch.tensor([0.5])], True, True)
            assert output == 'processed_output'


def test_prediction_pipeline_wav2vec(mock_wav2vec_model):
    """Test PredictionPipeline with Wav2vec model."""
    with patch.object(ModelFactory, 'create_model', return_value=mock_wav2vec_model):
        with patch.object(ps, 'Wav2vecPostprocessing') as MockPostprocessing:
            postprocessor_instance = MockPostprocessing.return_value
            postprocessor_instance.process = Mock(return_value='processed_output')
            
            pipeline = PredictionPipeline('wav2vec', 'config_path')
            assert pipeline.model == mock_wav2vec_model
            assert pipeline.postprocessing_strategy == postprocessor_instance
            
            output = pipeline.predict('input_data')
            mock_wav2vec_model.predict.assert_called_once_with('input_data')
            postprocessor_instance.process.assert_called_once_with([torch.tensor([0.9])], True, True)
            assert output == 'processed_output'


def test_prediction_pipeline_unknown_postprocessing():
    """Test PredictionPipeline with unknown postprocessing raises ValueError."""
    with patch.object(ModelFactory, 'create_model', return_value=Mock()):
        with pytest.raises(ValueError, match="Unknown postprocessing for model: unknown"):
            PredictionPipeline('unknown')


