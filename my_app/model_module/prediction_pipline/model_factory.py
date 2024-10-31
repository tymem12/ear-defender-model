from my_app.model_module.prediction_pipline import base_models as bm
from my_app.model_module.prediction_pipline import postprocessing_strategy as ps
from my_app.model_module.prediction_pipline import initialization_strategy as init_strat


class ModelFactory:
    _available_models = ['mesonet', 'wav2vec']

    @staticmethod
    def create_model(model_name, config_path=None):
        if model_name == 'mesonet':
            initialization_strategy = init_strat.MesonetInitialization()
            return bm.MesonetModel(config_path, initialization_strategy)
        elif model_name == 'wav2vec':
            initialization_strategy = init_strat.Wav2vecInitialization()
            return bm.Wav2wec(config_path, initialization_strategy)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
    @staticmethod
    def model_exists(model_name):
        return model_name in ModelFactory._available_models
    
    @staticmethod
    def get_available_models():
        return ModelFactory._available_models
    

class PredictionPipeline:
    def __init__(self, model_name, config_path=None, return_labels = True, return_scores = True):
        self.return_labels = return_labels
        self.return_scores = return_scores
        self.model = ModelFactory.create_model(model_name, config_path)
        self.postprocessing_strategy = self._get_postprocessing_strategy(model_name)
        self.model.initialized_model.eval()

    def _get_postprocessing_strategy(self, model_name):
        if model_name == 'wav2vec':
            return ps.Wav2vecPostprocessing(self.model.get_threshold_value())
        elif model_name == 'mesonet':
            return ps.MesoPostprocessing()
        else:
            raise ValueError(f"Unknown postprocessing for model: {model_name}")
        

    def predict(self, input_data):

        prediction = self.model.predict(input_data)
        model_output = self.postprocessing_strategy.process(prediction, self.return_scores, self.return_labels)
        return model_output
    

