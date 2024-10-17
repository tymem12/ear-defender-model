from my_app.model_module.models_manager import base_models as bm
from my_app.model_module.models_manager import postprocessing_strategy as ps
from my_app.model_module.models_manager import initialization_strategy as init_strat


class ModelFactory:
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


class PredictionPipeline:
    def __init__(self, model_name, config_path=None):
        self.model = ModelFactory.create_model(model_name, config_path)
        self.postprocessing_strategy = self._get_postprocessing_strategy(model_name)

    def _get_postprocessing_strategy(self, model_name):
        if model_name == 'modelA':
            return ps.Wav2vecPostprocessing()
        elif model_name == 'modelB':
            return ps.MesoPostprocessing()
        else:
            raise ValueError(f"Unknown postprocessing for model: {model_name}")
        

    def predict(self, input_data, return_labels = True):
        prediction = self.model.predict(input_data)
        if return_labels:
            postprocessed_output = self.postprocessing_strategy.process(prediction)
            return postprocessed_output
        return prediction
    
    def apply_postprocessing(self, prediction):
        postprocessed_output = self.postprocessing_strategy.process(prediction)
        return postprocessed_output