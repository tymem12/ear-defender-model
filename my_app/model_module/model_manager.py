from typing import Dict
import yaml
import os
import torch
import torch.nn as nn
from my_app.model_module.models.wav2vec.model import Model
from my_app.model_module.models.meso import meso_net

# later code below would need to change
from my_app.model_module.models_manager import mesonet_postprocessing, wav2vec_postprocessing


class ModelManager():
    __model_name :str = None
    __config_file: str = None
    __postprocessing_fun = None

    def __init__(self, model_name: str):
        self.model_name = model_name


    def initialize_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.model_name == "wav2vec":
            self.__config_file = os.getenv('WAV2VEC_CONFIG') if not self.__config_file else self.__config_file
            self.__postprocessing_fun = wav2vec_postprocessing
            config = load_config(self.__config_file)
            parameters = config['model']['parameters']
            checkpoint_path = config['checkpoint']['path']
            model = Model(args = parameters,device = device)
            model =nn.DataParallel(model).to(device)
            model.load_state_dict(torch.load(checkpoint_path,map_location=device))
            model.to(device)

            return model
                # return lcnn.FrontendLCNN(device=device, **config)
        # return None
        elif self.model_name  == "mesonet":
            self.__config_file = os.getenv('MESONET_CONFIG') if not self.__config_file else self.__config_file
            config = load_config(self.__config_file)
            checkpoint_path = config['checkpoint']['path']
            parameters = config['model']['parameters']
            self.__postprocessing_fun = mesonet_postprocessing


            model = meso_net.FrontendMesoInception4(
                input_channels=parameters.get("input_channels", 1),
                fc1_dim=parameters.get("fc1_dim", 1024),
                frontend_algorithm=parameters.get("frontend_algorithm", "lfcc"),
                device=device,
            )
            model.load_state_dict(torch.load(checkpoint_path))
            model = model.to(device)
            return model # mesonet_postprocessing
        else:
            raise ValueError(f"Model '{self.model_name}' not supported")

    @property
    def model_name(self) -> str:
        return self.__model_name
    

    @model_name.setter
    def model_name(self, model_name: str):
        self.__model_name = model_name

    @property
    def config_file(self) -> str:
        return self.__config_file
    

    @config_file.setter
    def config_file(self, config_file: str):
        self.__config_file = config_file
    
    @property
    def postprocessing_fun(self):
        return self.__postprocessing_fun
    

    @postprocessing_fun.setter
    def config_file(self, postprocessing_fun):
        self.__postprocessing_fun = postprocessing_fun


def load_config(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        return config