# This file includes code from the following repository:
# https://github.com/<repo-url>
# The code is related to the following paper:
# @inproceedings{kawa23b_interspeech,
#   author={Piotr Kawa and others},
#   title={{Improved DeepFake Detection Using Whisper Features}},
#   year=2023,
#   booktitle={Proc. INTERSPEECH 2023},
#   doi={10.21437/Interspeech.2023-1537}
# }

# file that returns the correct model based on the parameters

import os
import yaml
from typing import Dict
from model_module.models.meso import meso_net
from model_module.models.wav2vec.model import Model
import torch.nn as nn
import torch
import numpy as np

def mesonet_postprocessing(batch_pred):
    batch_pred = torch.sigmoid(batch_pred)
    batch_pred_label = (batch_pred + 0.5).int()
    return batch_pred_label

def wav2vec_postprocessing(batch_pred):
    # softmax = torch.nn.Softmax(dim=0)  # Softmax along the first dimension (for each tuple)
    elements = [ scor[1] for scor in batch_pred]
    label = [(tensor > float(os.getenv('WAV2VEC_EXPERIMENTAL_THRESHOLD'))).int()for tensor in elements ]
    return label

def get_model(model_name: str, config: Dict, device: str):
    if model_name == "wav2vec":
        checkpoint_path = 'pretrained/Best_LA_model_for_DF.pth'  # temporary
        model = Model(args = config,device = device)  # check if config is in correct format
        # nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        model =nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(checkpoint_path,map_location=device))
        model.to(device)

        return model, wav2vec_postprocessing
                # return lcnn.FrontendLCNN(device=device, **config)
        # return None
    elif model_name == "mesonet":
        checkpoint_path = 'pretrained/weights.pth'  # temporary

        model = meso_net.FrontendMesoInception4(
            input_channels=config.get("input_channels", 1),
            fc1_dim=config.get("fc1_dim", 1024),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
        model.load_state_dict(torch.load(checkpoint_path))
        model = model.to(device)
        return model, mesonet_postprocessing

    else:
        raise ValueError(f"Model '{model_name}' not supported")
    
# Load configurations from yaml files
def load_config(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        config = config['model']
    return config["parameters"]


if __name__ == '__main__':
    # Define the device (use GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load configuration files for both models
    meso_config = load_config('config_files/config_mesonet.yaml')
    wav2vec_config = load_config('config_files/config_wav2vec.yaml')

    # Test for MesoNet model
    try:
        print("Testing MesoNet model...")
        meso_model = get_model("mesonet", meso_config, device)
        
        # Print the number of parameters in the model
        num_params = sum(p.numel() for p in meso_model.parameters())
        print(f"MesoNet Model loaded with {num_params} parameters.")
        
        # Check if the model is on the correct device
        print(f"MesoNet Model is on device: {next(meso_model.parameters()).device}")

    except Exception as e:
        print(f"Failed to load MesoNet model: {e}")

    # Test for Wav2Vec model
    try:
        print("Testing Wav2Vec model...")
        wav2vec_model = get_model("wav2vec", wav2vec_config, device)
        
        # Print the number of parameters in the model
        num_params = sum(p.numel() for p in wav2vec_model.parameters())
        print(f"Wav2Vec Model loaded with {num_params} parameters.")
        
        # Check if the model is on the correct device
        print(f"Wav2Vec Model is on device: {next(wav2vec_model.parameters()).device}")

    except Exception as e:
        print(f"Failed to load Wav2Vec model: {e}")
