import pytest
from unittest.mock import patch, MagicMock
from my_app.model_module.prediction_pipline.model_factory import ModelFactory, PredictionPipeline
from my_app.model_module.prediction_pipline import base_models as bm
from my_app.model_module.prediction_pipline import postprocessing_strategy as ps
from my_app.model_module.prediction_pipline import initialization_strategy as init_strat

class TestModelFactory:
    
    @patch("my_app.model_module.prediction_pipline.model_factory.init_strat.MesonetInitialization")
    @patch("my_app.model_module.prediction_pipline.model_factory.bm.MesonetModel")
    def test_create_mesonet_model(self, MockMesonetModel, MockMesonetInitialization):
        # Mock initialization
        mock_init = MockMesonetInitialization.return_value
        mock_model_instance = MockMesonetModel.return_value
        
        model = ModelFactory.create_model('mesonet', config_path='path/to/config')
        
        MockMesonetInitialization.assert_called_once()
        MockMesonetModel.assert_called_once_with('path/to/config', mock_init)
        assert model == mock_model_instance

    @patch("my_app.model_module.prediction_pipline.model_factory.init_strat.Wav2vecInitialization")
    @patch("my_app.model_module.prediction_pipline.model_factory.bm.Wav2wec")
    def test_create_wav2vec_model(self, MockWav2wec, MockWav2vecInitialization):
        # Mock initialization
        mock_init = MockWav2vecInitialization.return_value
        mock_model_instance = MockWav2wec.return_value
        
        model = ModelFactory.create_model('wav2vec', config_path='path/to/config')
        
        MockWav2vecInitialization.assert_called_once()
        MockWav2wec.assert_called_once_with('path/to/config', mock_init)
        assert model == mock_model_instance

    def test_create_model_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown model: invalid_model"):
            ModelFactory.create_model('invalid_model')

    def test_model_exists(self):
        assert ModelFactory.model_exists('mesonet') is True
        assert ModelFactory.model_exists('wav2vec') is True
        assert ModelFactory.model_exists('invalid_model') is False

    def test_get_available_models(self):
        models = ModelFactory.get_available_models()
        assert models == ['mesonet', 'wav2vec']


class TestPredictionPipeline:

    @patch("my_app.model_module.prediction_pipline.model_factory.ModelFactory.create_model")
    @patch("my_app.model_module.prediction_pipline.model_factory.ps.MesoPostprocessing")
    def test_prediction_pipeline_mesonet(self, MockMesoPostprocessing, MockCreateModel):
        # Mock setup
        mock_model = MagicMock()
        MockCreateModel.return_value = mock_model
        mock_postprocessing = MockMesoPostprocessing.return_value
        
        pipeline = PredictionPipeline(model_name='mesonet', config_path='path/to/config')

        MockCreateModel.assert_called_once_with('mesonet', 'path/to/config')
        MockMesoPostprocessing.assert_called_once()
        assert pipeline.model == mock_model
        assert pipeline.postprocessing_strategy == mock_postprocessing

    @patch("my_app.model_module.prediction_pipline.model_factory.ModelFactory.create_model")
    @patch("my_app.model_module.prediction_pipline.model_factory.ps.Wav2vecPostprocessing")
    def test_prediction_pipeline_wav2vec(self, MockWav2vecPostprocessing, MockCreateModel):
        # Mock setup
        mock_model = MagicMock()
        mock_model.get_threshold_value.return_value = 0.5
        MockCreateModel.return_value = mock_model
        mock_postprocessing = MockWav2vecPostprocessing.return_value

        pipeline = PredictionPipeline(model_name='wav2vec', config_path='path/to/config')

        MockCreateModel.assert_called_once_with('wav2vec', 'path/to/config')
        MockWav2vecPostprocessing.assert_called_once_with(0.5)
        assert pipeline.model == mock_model
        assert pipeline.postprocessing_strategy == mock_postprocessing

    def test_prediction_pipeline_invalid_postprocessing(self):
        with pytest.raises(ValueError, match="Unknown model: invalid_model"):
            PredictionPipeline(model_name='invalid_model')

    @patch("my_app.model_module.prediction_pipline.model_factory.ModelFactory.create_model")
    @patch("my_app.model_module.prediction_pipline.model_factory.ps.MesoPostprocessing")
    def test_predict(self, MockMesoPostprocessing, MockCreateModel):
        # Mock setup
        mock_model = MagicMock()
        mock_model.predict.return_value = "mocked_prediction"
        MockCreateModel.return_value = mock_model
        mock_postprocessing = MockMesoPostprocessing.return_value
        mock_postprocessing.process.return_value = "processed_output"

        pipeline = PredictionPipeline(model_name='mesonet', config_path='path/to/config')
        result = pipeline.predict("mock_input_data")

        mock_model.predict.assert_called_once_with("mock_input_data")
        mock_postprocessing.process.assert_called_once_with("mocked_prediction", pipeline.return_scores, pipeline.return_labels)
        assert result == "processed_output"
