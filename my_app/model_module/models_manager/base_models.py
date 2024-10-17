from abc import ABC, abstractmethod
import os

# Base Model Interface
class Model(ABC):
    def __init__(self, config_path=None, initialization_strategy=None):
        if config_path is None:
            self.config_path = self.get_default_config()
        else:
            self.config_path = config_path
        self.initialization_strategy = initialization_strategy
        self.initialize_model()

    @abstractmethod
    def get_default_config(self):
        """Retrieve the default configuration file path."""
        pass

    def initialize_model(self):
        """Delegate the initialization to the strategy."""
        if self.initialization_strategy:
            self.initialization_strategy.initialize(self.config_path)
        else:
            raise ValueError("Initialization strategy is not provided.")

    @abstractmethod
    def predict(self, input_data):
        """Run the prediction on input data."""
        pass


class MesonetModel(Model):
    def get_default_config(self):
        return 'config_files/config_mesonet.yaml'

    def predict(self, input_data):
        return f"ModelA predictions for {input_data}"
    


class Wav2wec(Model):
    def get_default_config(self):
        return 'config_files/config_wav2vec.yaml'

    def predict(self, input_data):
        return f"ModelB predictions for {input_data}"
