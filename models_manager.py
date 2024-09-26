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



from typing import Dict
from models.meso import meso_net


def get_model(model_name: str, config: Dict, device: str):
    if model_name == "lcnn":
        # return lcnn.FrontendLCNN(device=device, **config)
        return None
    elif model_name == "mesonet":
        return meso_net.FrontendMesoInception4(
            input_channels=config.get("input_channels", 1),
            fc1_dim=config.get("fc1_dim", 1024),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported")