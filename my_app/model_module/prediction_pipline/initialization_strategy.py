from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from my_app.model_module.models.meso import meso_net
from my_app.model_module.models.wav2vec.model import Model
from my_app.utils import load_config


# Initialization Strategy Interface
class InitializationStrategy(ABC):

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = None
    @abstractmethod
    def initialize(self, config_path):
        pass

# Concrete Initialization Strategy for Mesonet Model
class MesonetInitialization(InitializationStrategy):
    def initialize(self, config_path):
        self.config = load_config(config_path)
        checkpoint_path = self.config['checkpoint']['path']
        self.parameters = self.config['model']['parameters']
        model = meso_net.FrontendMesoInception4(
            input_channels=self.parameters.get("input_channels", 1),
            fc1_dim=self.parameters.get("fc1_dim", 1024),
            frontend_algorithm=self.parameters.get("frontend_algorithm", "lfcc"),
            device=self.device,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model = model.to(self.device)
        # print(f"Initializing Mesonet model with config: {config_path}")
        return model

# Concrete Initialization Strategy for Wav2vec Model
class Wav2vecInitialization(InitializationStrategy):
    def initialize(self, config_path):
        self.config = load_config(config_path)
        self.parameters = self.config['model']['parameters']
        checkpoint_path = self.config['checkpoint']['path']
        model = Model(args=self.parameters, device=self.device)
        model = nn.DataParallel(model).to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device)
        # print(f"Initializing Wav2vec model with config: {config_path}")
        return model

